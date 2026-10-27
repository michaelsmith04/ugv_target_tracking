import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose, PoseArray

from dr_spaam.detector import Detector

import math

def fill_nan_gaps(scan: np.ndarray, max_gap: int, fill_value: float) -> np.ndarray:
    """
    Fill NaN runs in a 1D scan array.
    - Short runs (<= max_gap points) are linearly interpolated from
      their valid neighbors, producing a smooth transition instead of
      a hard depth discontinuity.
    - Long runs (> max_gap, i.e. likely genuine open space / out of
      range) are filled with fill_value (range_max).
    - Wraparound is NOT handled here; the scan is treated as linear.
      Edge NaN runs (touching index 0 or -1) always fall back to
      fill_value since there's no neighbor on one side to interpolate
      from.
    """
    scan = scan.copy()
    n = len(scan)
    is_nan = np.isnan(scan)
    if not is_nan.any():
        return scan

    idx = np.arange(n)
    i = 0
    while i < n:
        if not is_nan[i]:
            i += 1
            continue
        j = i
        while j < n and is_nan[j]:
            j += 1
        gap_len = j - i
        left_ok = i > 0
        right_ok = j < n
        if left_ok and right_ok and gap_len <= max_gap:
            left_val = scan[i - 1]
            right_val = scan[j]
            scan[i:j] = np.interp(idx[i:j], [i - 1, j], [left_val, right_val])
        else:
            scan[i:j] = fill_value
        i = j

    return scan

class DrSpaamNode(Node):
    def __init__(self):
        super().__init__('dr_spaam_node')

        self.declare_parameter('weight_path', '/home/ws/ugv_ws/src/ugv_main/ugv_human_detect/dr_spaam/2D_lidar_person_detection/dr_spaam/ckpts/dr_spaam_e40.pth')
        self.declare_parameter('conf_thresh', 0.4)
        self.declare_parameter('scan_topic', '/scan_resampled')
        self.declare_parameter('laser_fov_deg', 360.0)
        self.declare_parameter('nan_fill_max_gap', 0)
        self._expected_len = None
        self._canonical_angles = None

        weight_path = self.get_parameter('weight_path').value
        self.conf_thresh = self.get_parameter('conf_thresh').value
        scan_topic = self.get_parameter('scan_topic').value
        laser_fov_deg = self.get_parameter('laser_fov_deg').value
        self.nan_fill_max_gap = self.get_parameter('nan_fill_max_gap').value

        self.detector = Detector(
            weight_path,
            model="DR-SPAAM",
            gpu=True,
            stride=1,
            panoramic_scan=True,
        )
        self.detector.set_laser_fov(laser_fov_deg)

        self.scan_sub = self.create_subscription(LaserScan, scan_topic, self.scan_callback, 10)
        self.detections_pub = self.create_publisher(PoseArray, '/dr_spaam_detections', 10)

        self.get_logger().info(f'DR-SPAAM node ready, subscribed to {scan_topic}')

    def scan_callback(self, msg: LaserScan):
        scan = np.array(msg.ranges, dtype=np.float32)

        range_max = msg.range_max if msg.range_max > 0.0 else 25.0
        # Treat below-range_min readings as invalid too, same as NaN/inf.
        scan[~np.isfinite(scan)] = np.nan
        scan[scan <= msg.range_min] = np.nan

        scan = fill_nan_gaps(scan, self.nan_fill_max_gap, range_max)

        if len(scan) < 10:
            self.get_logger().warn(f'Skipping malformed scan (len={len(scan)})')
            return

        # This lidar's point count varies every revolution (motor RPM
        # jitter -> angle_increment recomputed as 2*pi/N each scan), so
        # index i doesn't correspond to a fixed angle across messages.
        # Resample onto a fixed angular grid, captured once from the
        # first scan, before handing anything to the detector.
        raw_angles = msg.angle_min + np.arange(len(scan), dtype=np.float32) * msg.angle_increment

        if self._expected_len is None:
            self._expected_len = len(scan)
            self._canonical_angles = raw_angles.copy()
            self.get_logger().info(
                f'Locking canonical scan grid: {self._expected_len} points'
            )

        scan = np.interp(
            self._canonical_angles, raw_angles, scan,
            left=range_max, right=range_max,
        ).astype(np.float32)

        dets_xy, dets_cls, _ = self.detector(scan)
        dets_cls = dets_cls.flatten()

        pose_array = PoseArray()
        pose_array.header = msg.header

        ANGULAR_CORRECTION_RAD = math.radians(-90)  # <-- fill in your measured offset
        for (x, y), conf in zip(dets_xy, dets_cls):
            if conf < self.conf_thresh:
                continue

            # forward/lateral swap (established last round)
            fx, fy = float(y), -float(x)

            # apply fixed rotational correction
            cos_a, sin_a = math.cos(ANGULAR_CORRECTION_RAD), math.sin(ANGULAR_CORRECTION_RAD)
            corrected_x = fx * cos_a - fy * sin_a
            corrected_y = fx * sin_a + fy * cos_a

            pose = Pose()
            pose.position.x = -corrected_x
            pose.position.y = -corrected_y
            pose.position.z = 0.0
            pose_array.poses.append(pose)

        self.detections_pub.publish(pose_array)

def main(args=None):
    rclpy.init(args=args)
    node = DrSpaamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()