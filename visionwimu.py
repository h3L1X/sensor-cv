import asyncio
import json
import math
import struct
import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs
from bleak import BleakClient, BleakScanner
from ultralytics import YOLO

NOTIFY_UUID = "0000ffe4-0000-1000-8000-00805f9a34fb"
ACC_SCALE = 16.0 / 32768
GYRO_SCALE = 2000.0 / 32768
ANGLE_SCALE = 180.0 / 32768
STALE_AFTER_S = 1.0  # IMU data older than this is flagged as stale


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


class IMUReader:
    """Reads the WitMotion BLE sensor on a background thread.

    Latest values are kept in a lock-protected dict so the camera loop
    can grab them at any time without touching asyncio.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._latest = None          # dict of latest sample
        self._yaw_offset = 0.0
        self._buf = bytearray()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.status = "starting"

    # ---------- public API (called from the main thread) ----------
    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)

    def zero_yaw(self):
        with self._lock:
            if self._latest is not None:
                self._yaw_offset = self._latest["raw_yaw"]

    def snapshot(self):
        """Return the latest IMU reading (with age), or None if nothing yet."""
        with self._lock:
            if self._latest is None:
                return None
            s = dict(self._latest)
            yaw_offset = self._yaw_offset
        age = time.time() - s["t"]
        return {
            "roll_deg": round(s["roll"], 2),
            "pitch_deg": round(s["pitch"], 2),
            "yaw_deg": round(wrap180(s["raw_yaw"] - yaw_offset), 2),
            "gyro_dps": [round(v, 2) for v in s["gyro"]],
            "accel_g": [round(v, 3) for v in s["acc"]],
            "age_seconds": round(age, 3),
            "stale": age > STALE_AFTER_S,
        }

    # ---------- BLE internals (background thread) ----------
    def _on_data(self, _, data: bytearray):
        buf = self._buf
        buf.extend(data)
        while len(buf) >= 2:
            if buf[0] != 0x55:
                del buf[0]
                continue
            if buf[1] == 0x61:
                if len(buf) < 20:
                    return
                ax, ay, az, gx, gy, gz, roll, pitch, yaw = struct.unpack(
                    "<9h", bytes(buf[2:20])
                )
                sample = {
                    "t": time.time(),
                    "acc": (ax * ACC_SCALE, ay * ACC_SCALE, az * ACC_SCALE),
                    "gyro": (gx * GYRO_SCALE, gy * GYRO_SCALE, gz * GYRO_SCALE),
                    "roll": roll * ANGLE_SCALE,
                    "pitch": pitch * ANGLE_SCALE,
                    "raw_yaw": yaw * ANGLE_SCALE,
                }
                with self._lock:
                    self._latest = sample
                del buf[:20]
            else:
                del buf[0]

    async def _connect_loop(self):
        # Reconnects automatically if the sensor drops out
        while not self._stop.is_set():
            self.status = "scanning"
            device = await BleakScanner.find_device_by_filter(
                lambda d, ad: (d.name or "").startswith("WT"), timeout=10
            )
            if device is None:
                self.status = "not found (retrying)"
                continue

            self.status = f"connecting to {device.name}"
            self._buf.clear()
            try:
                async with BleakClient(device) as client:
                    await client.start_notify(NOTIFY_UUID, self._on_data)
                    self.status = f"connected ({device.name})"
                    while not self._stop.is_set() and client.is_connected:
                        await asyncio.sleep(0.1)
                    if client.is_connected:
                        await client.stop_notify(NOTIFY_UUID)
            except Exception as e:
                self.status = f"error: {e}"
                await asyncio.sleep(2)

        self.status = "stopped"

    def _run(self):
        asyncio.run(self._connect_loop())


def get_robust_3d_data(depth_frame_array, intrinsics, bbox):
    """Calculates the robust 3D coordinate and distance of an object."""
    xmin, ymin, xmax, ymax = map(int, bbox)
    roi = depth_frame_array[ymin:ymax, xmin:xmax]

    valid_depths = roi[roi > 0]
    if len(valid_depths) < 10:
        return None, None

    # Robust center using interquartile range to ignore background/noise
    p25 = np.percentile(valid_depths, 25)
    p75 = np.percentile(valid_depths, 75)
    core = valid_depths[(valid_depths >= p25) & (valid_depths <= p75)]
    center_z = np.median(core) if len(core) > 0 else np.median(valid_depths)

    center_x = (xmin + xmax) // 2
    center_y = (ymin + ymax) // 2

    point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [center_x, center_y], center_z)
    return point_3d, center_z


def calculate_3d_distance(pt1, pt2):
    """Calculates Euclidean distance between two 3D points in meters."""
    return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2 + (pt1[2] - pt2[2])**2)


def main():
    print("Loading YOLO model...")
    model = YOLO("yolov8n.pt")

    # Start IMU in the background
    imu = IMUReader()
    imu.start()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    cv2.namedWindow("Crane LLM Vision", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Crane LLM Vision", 1280, 960)

    print("\nSystem Ready.")
    print("Press 's' to take a snapshot and generate the LLM JSON.")
    print("Press 'z' to zero the IMU yaw.")
    print("Press 'q' to quit.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)

            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            depth_array = np.asanyarray(depth_frame.get_data()) * depth_frame.get_units()
            color_image = np.asanyarray(color_frame.get_data())

            results = model(color_image, stream=True, verbose=False)

            detected_objects = []

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls_id]

                    cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(color_image, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    detected_objects.append({
                        "label": label,
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2]
                    })

            # On-screen overlay: instructions + live IMU readout
            cv2.putText(color_image, "s: snapshot | z: zero yaw | q: quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            imu_now = imu.snapshot()
            if imu_now:
                imu_text = (f"IMU r {imu_now['roll_deg']:.1f} p {imu_now['pitch_deg']:.1f} "
                            f"y {imu_now['yaw_deg']:.1f}" + ("  (STALE)" if imu_now["stale"] else ""))
                color = (0, 0, 255) if imu_now["stale"] else (255, 200, 0)
            else:
                imu_text = f"IMU: {imu.status}"
                color = (0, 0, 255)
            cv2.putText(color_image, imu_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("Crane LLM Vision", color_image)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('z'):
                imu.zero_yaw()
                print("IMU yaw zeroed.")

            # --- SNAPSHOT TRIGGER ---
            elif key == ord('s'):
                print("\nCapturing snapshot...")

                llm_payload = {
                    "timestamp": time.time(),
                    "crane_imu": imu.snapshot(),   # None if sensor never connected
                    "environment_objects": [],
                    "spatial_relationships": []
                }

                if llm_payload["crane_imu"] is None:
                    print("Warning: no IMU data yet (" + imu.status + ")")
                elif llm_payload["crane_imu"]["stale"]:
                    print("Warning: IMU data is stale.")

                valid_3d_objects = []

                for i, obj in enumerate(detected_objects):
                    pt_3d, dist_to_cam = get_robust_3d_data(depth_array, intrinsics, obj["bbox"])

                    if pt_3d:
                        obj_data = {
                            "id": f"{obj['label']}_{i}",
                            "type": obj["label"],
                            "camera_distance_meters": round(dist_to_cam, 2),
                            "coordinates_3d": [round(c, 2) for c in pt_3d]
                        }
                        llm_payload["environment_objects"].append(obj_data)
                        valid_3d_objects.append(obj_data)

                for i in range(len(valid_3d_objects)):
                    for j in range(i + 1, len(valid_3d_objects)):
                        obj_A = valid_3d_objects[i]
                        obj_B = valid_3d_objects[j]

                        dist = calculate_3d_distance(obj_A["coordinates_3d"], obj_B["coordinates_3d"])

                        llm_payload["spatial_relationships"].append({
                            "object_1": obj_A["id"],
                            "object_2": obj_B["id"],
                            "distance_apart_meters": round(dist, 2)
                        })

                with open("llm_payload.json", "w") as f:
                    json.dump(llm_payload, f, indent=4)

                print("Snapshot saved to 'llm_payload.json'!")

                flash = np.ones_like(color_image) * 255
                cv2.imshow("Crane LLM Vision", flash)
                cv2.waitKey(100)

            elif key == ord('q'):
                break

    finally:
        imu.stop()
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()