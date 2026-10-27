#!/usr/bin/env python3
"""
Converts OAK-D-LITE's vision_msgs/Detection3DArray spatial detections into a
single geometry_msgs/PoseWithCovarianceStamped for robot_localization's
ukf_node.

IMPORTANT things this node does NOT do for you yet, flagged deliberately:
  - It does not verify units. Confirm on real hardware whether position.z is
    in meters or millimeters (echo a detection at a known 2 m distance) and
    set the 'depth_unit_is_mm' parameter accordingly.
  - The driver reports an all-zero covariance on every detection. This node
    OVERRIDES it with a simple distance-scaled placeholder model
    (var_z = k_depth * range^2) rather than trusting the zeros. k_depth and
    var_xy_base are PLACEHOLDERS until you calibrate against known distances.
  - Detection selection defaults to the first entry in the array. Extend
    select_detection() once you need multi-person handling or want to filter
    by hypothesis.class_id / score.

LOGGING (added for the static grid test):
  - oak_detections_<time>.csv : one row per detection in every received
    Detection3DArray (ALL classes, before any filtering), or one row with
    n_det=0 for an empty array. 'selected' marks the detection this node
    actually publishes.
  - oak_frames_<time>.csv     : one row per camera frame, from the topic in
    'frame_topic'. Needed so frames with no detection are counted as misses
    (the detection topic may not publish at all when nothing is detected).
  - Node/visit labels come from std_msgs/String on 'marker_topic':
        'start A05 1'  -> label rows with node A05, visit 1
        'stop'         -> label rows as idle
"""

import csv
import os
from datetime import datetime

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import String
from sensor_msgs.msg import CameraInfo

# --- LOGGING: column definitions -------------------------------------------
DET_COLS = [
    'recv_ns', 'stamp_ns', 'frame_id', 'state', 'node_id', 'visit_id',
    'n_det', 'det_idx', 'selected', 'class_id', 'score',
    'x_raw', 'y_raw', 'z_raw', 'x_m', 'y_m', 'z_m',
    'bbox_cx', 'bbox_cy', 'bbox_w', 'bbox_h', 'var_z_model',
]
FRAME_COLS = ['recv_ns', 'stamp_ns', 'state', 'node_id', 'visit_id']


class OakPoseConverter(Node):
    def __init__(self):
        super().__init__('oak_pose_converter')

        # --- Placeholder covariance model parameters (tune via calibration) ---
        self.declare_parameter('var_xy_base', 0.02)   # m^2, lateral noise floor
        self.declare_parameter('k_depth', 0.01)        # depth_var = k * range^2
        self.declare_parameter('var_orientation', 1.0e3)  # unused, masked

        self.declare_parameter('depth_unit_is_mm', False)  # VERIFY on hardware
        self.declare_parameter('class_id_filter', '')     # '' = no filtering
        self.declare_parameter('min_score', 0.0)

        self.declare_parameter('input_topic', '/oak/nn/spatial_detections')
        self.declare_parameter('output_topic', '/target/oak/pose_gated')

        # --- LOGGING: parameters ---
        self.declare_parameter('log_enabled', False)
        self.declare_parameter('log_dir', '~/oak_logs')
        # A topic that publishes once per camera frame (CameraInfo). Leave ''
        # to disable the frame log. Check names with `ros2 topic list`.
        self.declare_parameter('frame_topic', '/oak/rgb/camera_info')
        self.declare_parameter('marker_topic', '/eval/marker')

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value

        self.sub = self.create_subscription(
            Detection3DArray, input_topic, self.callback, 10)
        self.pub = self.create_publisher(
            PoseWithCovarianceStamped, output_topic, 10)

        # --- LOGGING: setup ---
        self._label_state = 'idle'
        self._label_node = ''
        self._label_visit = ''
        self._det_f = self._det_w = None
        self._frame_f = self._frame_w = None

        if self.get_parameter('log_enabled').value:
            log_dir = os.path.expanduser(self.get_parameter('log_dir').value)
            os.makedirs(log_dir, exist_ok=True)
            tag = datetime.now().strftime('%Y%m%d_%H%M%S')

            det_path = os.path.join(log_dir, f'oak_detections_{tag}.csv')
            self._det_f = open(det_path, 'w', newline='')
            self._det_w = csv.writer(self._det_f)
            self._det_w.writerow(DET_COLS)
            self.get_logger().info(f'logging detections to {det_path}')

            frame_topic = self.get_parameter('frame_topic').value
            if frame_topic:
                frame_path = os.path.join(log_dir, f'oak_frames_{tag}.csv')
                self._frame_f = open(frame_path, 'w', newline='')
                self._frame_w = csv.writer(self._frame_f)
                self._frame_w.writerow(FRAME_COLS)
                self.frame_sub = self.create_subscription(
                    CameraInfo, frame_topic, self.frame_callback, 10)
                self.get_logger().info(f'logging frame stamps to {frame_path}')
            else:
                self.get_logger().warn(
                    "frame_topic not set: frames with NO detection cannot be "
                    "counted, so recall will be unmeasurable.")

            self.marker_sub = self.create_subscription(
                String, self.get_parameter('marker_topic').value,
                self.marker_callback, 10)

        self.get_logger().info(
            f'oak_pose_converter: {input_topic} -> {output_topic}')

    # --- LOGGING: helpers ---------------------------------------------------
    def marker_callback(self, msg: String):
        """'start <node_id> <visit_id>' or 'stop'."""
        parts = msg.data.split()
        if len(parts) >= 3 and parts[0] == 'start':
            self._label_state = 'rec'
            self._label_node, self._label_visit = parts[1], parts[2]
        elif parts and parts[0] == 'stop':
            self._label_state, self._label_node, self._label_visit = 'idle', '', ''
        else:
            self.get_logger().warn(f"unrecognised marker '{msg.data}'")
            return
        self.get_logger().info(
            f'marker: {self._label_state} {self._label_node} {self._label_visit}')

    def frame_callback(self, msg: CameraInfo):
        stamp_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        self._frame_w.writerow([
            self.get_clock().now().nanoseconds, stamp_ns,
            self._label_state, self._label_node, self._label_visit])
        self._frame_f.flush()

    def log_detections(self, msg: Detection3DArray, selected_idx):
        """Log every detection in the array, or one empty row if none."""
        if self._det_w is None:
            return
        recv_ns = self.get_clock().now().nanoseconds
        stamp_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        scale = 1000.0 if self.get_parameter('depth_unit_is_mm').value else 1.0
        k_depth = self.get_parameter('k_depth').value
        n_det = len(msg.detections)
        base = [recv_ns, stamp_ns, msg.header.frame_id,
                self._label_state, self._label_node, self._label_visit, n_det]

        if n_det == 0:
            self._det_w.writerow(base + [''] * 15)
        for i, det in enumerate(msg.detections):
            if len(det.results) == 0:
                self._det_w.writerow(base + [i, 0] + [''] * 13)
                continue
            r = det.results[0]
            p = r.pose.pose.position
            x_m, y_m, z_m = p.x / scale, p.y / scale, p.z / scale
            var_z = k_depth * max(abs(z_m), 0.1) ** 2
            bb = det.bbox
            self._det_w.writerow(base + [
                i, int(i == selected_idx),
                r.hypothesis.class_id, r.hypothesis.score,
                p.x, p.y, p.z, x_m, y_m, z_m,
                bb.center.position.x, bb.center.position.y,
                bb.size.x, bb.size.y, var_z])
        self._det_f.flush()

    def destroy_node(self):
        for f in (self._det_f, self._frame_f):
            if f is not None:
                f.close()
        super().destroy_node()

    # ------------------------------------------------------------------------
    def select_detection(self, msg: Detection3DArray):
        """Pick which detection to track. Single-person placeholder: first
        detection that (optionally) passes class_id/score filters.
        Returns (index, result), or (None, None)."""
        class_filter = self.get_parameter('class_id_filter').value
        min_score = self.get_parameter('min_score').value

        for i, det in enumerate(msg.detections):
            if len(det.results) == 0:
                continue
            result = det.results[0]
            if class_filter and result.hypothesis.class_id != class_filter:
                continue
            if result.hypothesis.score < min_score:
                continue
            return i, result

        return None, None

    def callback(self, msg: Detection3DArray):
        idx, result = self.select_detection(msg)

        # --- LOGGING: log before any early return, so misses are recorded ---
        self.log_detections(msg, idx)

        if result is None:
            return

        pos = result.pose.pose.position
        x, y, z = pos.x, pos.y, pos.z

        if self.get_parameter('depth_unit_is_mm').value:
            x, y, z = x / 1000.0, y / 1000.0, z / 1000.0

        out = PoseWithCovarianceStamped()
        out.header.stamp = msg.header.stamp  # preserve original detection time
        out.header.frame_id = msg.header.frame_id

        out.pose.pose.position.x = x
        out.pose.pose.position.y = y
        out.pose.pose.position.z = z
        out.pose.pose.orientation.w = 1.0  # unused, masked in UKF config

        var_xy_base = self.get_parameter('var_xy_base').value
        k_depth = self.get_parameter('k_depth').value
        var_o = self.get_parameter('var_orientation').value

        # Range used for the depth-noise model. Using the raw z (depth) value
        # is a simplification -- fine for a forward-facing target, but revisit
        # if the target moves far off the optical axis.
        range_m = max(abs(z), 0.1)
        var_z = k_depth * (range_m ** 2)

        cov = [0.0] * 36
        cov[0] = var_xy_base       # x
        cov[7] = var_xy_base       # y
        cov[14] = var_z            # z (distance-scaled)
        cov[21] = var_o
        cov[28] = var_o
        cov[35] = var_o
        out.pose.covariance = cov

        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = OakPoseConverter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()