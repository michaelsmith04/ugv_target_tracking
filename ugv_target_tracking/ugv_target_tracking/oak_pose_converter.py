#!/usr/bin/env python3
"""
Converts OAK-D-LITE's vision_msgs/Detection3DArray spatial detections into a
single geometry_msgs/PoseWithCovarianceStamped for robot_localization's
ukf_node.

COVARIANCE / CORRECTIONS (from the static grid test)
  The driver reports an all-zero covariance, so this node builds one from a model
  fitted on held-out residuals of the static grid test. The model describes the
  CORRECTED measurement, so the same corrections are applied first:

    1. depth:    z_c = poly(z_raw)            (quadratic, 'depth_poly')
                 x_c = (x/z) * z_c, y_c = (y/z) * z_c   (keeps the measured bearing)
    2. bearing:  rotate about the camera's vertical axis by -'bearing_offset_deg'
    3. covariance in the camera frame (x right, y down, z forward):
         sigma_z^2  = (depth_c * z_c^2)^2 + depth_s0^2
         sigma_x^2  = (z_c * sigma_px / fx)^2 + (x/z)^2 sigma_z^2 + lat_s0^2
         sigma_px   = lat_px_per_bbox_px * (bbox_scale_x * bbox_w)   [px, full image]
         sigma_xz   = (x/z) * sigma_z^2
         sigma_y^2  = var_y            (NOT covered by the test; kept from the old placeholder)

  PROVISIONAL: the depth polynomial and depth_c compensate a stereo disparity offset
  of about 1.2 px found in the grid test. If the stereo calibration is fixed, set
  depth_poly to [0, 1, 0] and refit depth_c / depth_s0 from a new grid test.
  Measurements whose raw depth is outside [depth_meas_min, depth_meas_max] (the tested
  range) are not published when gate_estimates is true.

Other things this node does NOT do for you:
  - It does not verify units. Confirm on real hardware whether position.z is
    in meters or millimeters (echo a detection at a known 2 m distance) and
    set the 'depth_unit_is_mm' parameter accordingly.
  - Detection selection defaults to the first entry in the array. Extend
    select_detection() once you need multi-person handling or want to filter
    by hypothesis.class_id / score.
  - 'use_fitted_cov:=false' restores the old placeholder model
    (var_z = k_depth * range^2, var_xy = var_xy_base) with no corrections.

LOGGING (added for the static grid test):
  - oak_detections_<time>.csv : one row per detection in every received
    Detection3DArray (ALL classes, before any filtering), or one row with
    n_det=0 for an empty array. 'selected' marks the detection this node
    actually publishes. x_raw..z_m are RAW sensor values; var_z_model is the
    model variance at the corrected depth.
  - oak_frames_<time>.csv     : one row per camera frame, from the topic in
    'frame_topic'. Needed so frames with no detection are counted as misses
    (the detection topic may not publish at all when nothing is detected).
  - Node/visit labels come from std_msgs/String on 'marker_topic':
        'start A05 1'  -> label rows with node A05, visit 1
        'stop'         -> label rows as idle
"""

import csv
import math
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

        # --- Legacy placeholder model (used only when use_fitted_cov is false) ---
        self.declare_parameter('var_xy_base', 0.02)   # m^2, lateral noise floor
        self.declare_parameter('k_depth', 0.01)        # depth_var = k * range^2
        self.declare_parameter('var_orientation', 1.0e3)  # unused, masked

        # --- Fitted covariance model + corrections (static grid test) ---
        self.declare_parameter('use_fitted_cov', True)
        self.declare_parameter('apply_corrections', True)
        self.declare_parameter('gate_estimates', True)
        # z_true = a2*z^2 + a1*z + a0, fitted on node-median depths 1.12-7.77 m raw
        self.declare_parameter('depth_poly',
                               [0.05327034660022406, 0.981667733711408, -0.2983758721293896])
        self.declare_parameter('depth_meas_min', 0.5)   # m, raw depth range covered by the test
        self.declare_parameter('depth_meas_max', 7.8)
        self.declare_parameter('bearing_offset_deg', 1.1)  # measured bearing is this far to the right
        self.declare_parameter('depth_c', 0.0085)       # 1/m, inflated working value (fit: 0.0054)
        self.declare_parameter('depth_s0', 0.02)        # m, depth floor (judgement)
        self.declare_parameter('lat_px_per_bbox_px', 0.0058)
        self.declare_parameter('lat_px_default', 0.6)   # px, used if the bbox width is missing
        self.declare_parameter('lat_s0', 0.05)          # m, lateral floor (judgement)
        self.declare_parameter('fx', 912.44)            # px, RGB intrinsics used by the analysis
        self.declare_parameter('bbox_scale_x', 2.0)     # preview px -> full-image px
        self.declare_parameter('var_y', 0.02)           # m^2, vertical axis (not covered by the test)

        self.declare_parameter('depth_unit_is_mm', False)  # VERIFY on hardware
        self.declare_parameter('class_id_filter', '')     # '' = no filtering
        self.declare_parameter('min_score', 0.5)

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
            f'oak_pose_converter: {input_topic} -> {output_topic} '
            f"(fitted covariance: {self.get_parameter('use_fitted_cov').value})")

    # --- covariance model + corrections -------------------------------------
    def _p(self, name):
        return self.get_parameter(name).value

    def correct_depth(self, z):
        if not self._p('apply_corrections'):
            return z
        a2, a1, a0 = self._p('depth_poly')
        z_poly = (a2 * z + a1) * z + a0
        lo, hi = 0.5, 1.1           # correction fades in over 0.5-1.1 m
        if z >= hi:
            return z_poly
        return z + max(0.0, (z - lo) / (hi - lo)) * (z_poly - z)

    def raw_depth_in_range(self, z):
        return self._p('depth_meas_min') <= z <= self._p('depth_meas_max')

    def correct_position(self, x, y, z):
        """Raw (x, y, z) [m, camera optical frame] -> corrected (x, y, z).
        Depth is corrected along the measured ray, then the constant bearing offset is removed."""
        if not self._p('apply_corrections'):
            return x, y, z
        z_c = self.correct_depth(z)
        x_d, y_d = (x / z) * z_c, (y / z) * z_c
        rho = math.hypot(x_d, z_c)
        psi = math.atan2(x_d, z_c) - math.radians(self._p('bearing_offset_deg'))
        return rho * math.sin(psi), y_d, rho * math.cos(psi)

    def sigma_z(self, z_c):
        c, s0 = self._p('depth_c'), self._p('depth_s0')
        return math.sqrt((c * z_c * z_c) ** 2 + s0 ** 2)

    def fitted_covariance(self, x, y, z, bbox_w_preview):
        """Covariance entries (cxx, cxz, czz, cyy) for a CORRECTED position (camera frame)."""
        z = max(z, 0.1)
        var_z = self.sigma_z(z) ** 2
        ratio = x / z
        if bbox_w_preview and bbox_w_preview > 0:
            sig_px = self._p('lat_px_per_bbox_px') * self._p('bbox_scale_x') * bbox_w_preview
        else:
            sig_px = self._p('lat_px_default')
        var_x = ((z * sig_px / self._p('fx')) ** 2 + ratio ** 2 * var_z + self._p('lat_s0') ** 2)
        return var_x, ratio * var_z, var_z, self._p('var_y')

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
        use_fit = self._p('use_fitted_cov')
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
            if use_fit and z_m > 0:
                var_z = self.sigma_z(self.correct_depth(z_m)) ** 2   # model variance at corrected depth
            else:
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
    def select_detection(self, msg):
        class_filter = self.get_parameter('class_id_filter').value
        min_score = self.get_parameter('min_score').value
        best_i, best_r, best_s = None, None, -1.0

        allowed_classes = ['0', '2'] # Left and right legs

        for i, det in enumerate(msg.detections):
            if len(det.results) == 0:
                continue
            r = det.results[0]
            if class_filter and r.hypothesis.class_id != class_filter:
                continue
            if r.hypothesis.score < min_score:
                continue
            if r.hypothesis.score > best_s:
                best_i, best_r, best_s = i, r, r.hypothesis.score
        return best_i, best_r

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

        use_fit = self._p('use_fitted_cov')
        if use_fit:
            if z <= 0.0:
                return   # invalid depth: do not feed the filter a zero-range measurement
            if self._p('gate_estimates') and self._p('apply_corrections') \
                    and not self.raw_depth_in_range(z):
                return   # outside the depth range covered by the grid test
            x, y, z = self.correct_position(x, y, z)

        out = PoseWithCovarianceStamped()
        out.header.stamp = msg.header.stamp  # preserve original detection time
        out.header.frame_id = msg.header.frame_id

        out.pose.pose.position.x = x
        out.pose.pose.position.y = y
        out.pose.pose.position.z = z
        out.pose.pose.orientation.w = 1.0  # unused, masked in UKF config

        var_o = self.get_parameter('var_orientation').value
        cov = [0.0] * 36

        if use_fit:
            bbox_w = msg.detections[idx].bbox.size.x
            cxx, cxz, czz, cyy = self.fitted_covariance(x, y, z, bbox_w)
            cov[0] = cxx       # x
            cov[2] = cxz       # x-z
            cov[12] = cxz      # z-x
            cov[7] = cyy       # y
            cov[14] = czz      # z
        else:
            var_xy_base = self.get_parameter('var_xy_base').value
            k_depth = self.get_parameter('k_depth').value
            range_m = max(abs(z), 0.1)
            cov[0] = var_xy_base
            cov[7] = var_xy_base
            cov[14] = k_depth * (range_m ** 2)
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