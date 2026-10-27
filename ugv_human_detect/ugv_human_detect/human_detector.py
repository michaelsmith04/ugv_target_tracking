import csv
import json
import math
import os
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import JointState, CompressedImage
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from geometry_msgs.msg import PoseWithCovarianceStamped

from ultralytics import YOLO
import torchvision

_orig_nms = torchvision.ops.nms


def _cpu_nms(boxes, scores, iou_threshold):
    keep = _orig_nms(boxes.cpu().float(), scores.cpu().float(), iou_threshold)
    return keep.to(boxes.device)

torchvision.ops.nms = _cpu_nms
torchvision.ops.boxes.nms = _cpu_nms

PERSON_CLASS_ID = 0                # COCO "person"
LEFT_ANKLE, RIGHT_ANKLE = 15, 16   # COCO keypoint indices

# --- LOGGING: column definitions ------------------------------------------------------------------------
DET_COLS = ['recv_ns', 'stamp_ns', 'state', 'node_id', 'visit_id', 'n_det', 'det_idx', 'conf',
            'x1', 'y1', 'x2', 'y2', 'kL_u', 'kL_v', 'kL_c', 'kR_u', 'kR_v', 'kR_c',
            'infer_ms', 'img_w', 'img_h']
POSE_COLS = ['recv_ns', 'stamp_ns', 'state', 'node_id', 'visit_id', 'status', 'ready', 'n_dets',
             'pan_deg', 'tilt_raw_deg', 'pan_rate_dps', 'tilt_rate_dps',
             'ext_ok', 'ext_x', 'ext_y', 'ext_cxx', 'ext_cxy', 'ext_cyy', 'ext_d', 'ext_sig_r', 'ext_det',
             'gnd_ok', 'gnd_x', 'gnd_y', 'gnd_cxx', 'gnd_cxy', 'gnd_cyy', 'gnd_r', 'gnd_sig_r', 'gnd_det']


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def stamp_to_ns(stamp):
    return stamp.sec * 10 ** 9 + stamp.nanosec


def _n(v):
    """Compact float formatting for the CSV; None -> empty."""
    return '' if v is None else f'{float(v):.6g}'


class HumanDetector(Node):
    def __init__(self):
        super().__init__('human_detector')

        defaults = {
            'image_topic': '/image_rect/compressed',
            'detections_topic': '/pantilt/detections',
            'poses_topic': '/pantilt/pose_estimates',   # full-body (extent) estimate; '<topic>_ground' = ankle estimate
            'model_path': '/home/ws/ugv_ws/src/ugv_main/ugv_human_detect/models/yolov8n-pose.pt',
            'confidence_threshold': 0.5,
            'device': 'cuda:0',
            # joints in /ugv/joint_states
            'pan_joint': 'pt_base_link_to_pt_link1',
            'tilt_joint': 'pt_link1_to_pt_link2',
            # conventions used below: pan +ve = CCW/left, tilt +ve = camera looking UP (radians)
            'pan_sign': -1.0,
            # pan_true = pan_sign * reading + pan_offset. Stationary tests showed bearings ~17 deg too far left.
            # Static grid test (pan_offset = pi - 0.1618): a person on the marker axis appeared 13.2 deg to the
            # RIGHT, so the estimates must be rotated +13.2 deg CCW. If the sign is wrong the on-axis bearing in the
            # next grid test will double to ~26 deg: then use math.pi - 0.1618 - math.radians(13.20).
            # NOTE: this is the TOTAL constant offset (servo zero + rig alignment + intrinsics/cx mismatch).
            'pan_offset': math.pi - 0.1618 + math.radians(13.20),
            'tilt_sign': 1.0,
            # tilt_true = tilt_reading + tilt_offset (static). Fitted by the CALIB log line.
            'tilt_offset': -0.0586,
            # The joint readings describe the pose joint_lag seconds BEFORE they arrive: the pose at image time t
            # is reading(t + joint_lag). Pose output is delayed until that reading exists. Fitted by CALIB.
            'joint_lag': 0.0,
            # geometry
            'output_frame': 'pt_base_link',
            'pan_axis_x': 0.010151,
            'pan_axis_y': 0.0,
            'cam_base_height': 0.22,   # optical centre height at tilt = 0 (m)
            'cam_arm': 0.065,          # camera lever arm about the tilt axis (m)
            'ankle_height': 0.09, # Ankle height in m
            # person model for the full-body ("extent") estimate: SET THIS TO THE REAL HEIGHT OF THE PERSON YOU FOLLOW
            'person_height': 1.7,
            'sigma_person_height': 0.08,   # ~0.02-0.03 if you enrol the person's measured height
            'sigma_box_pix': 5.0,          # bbox top/bottom edge noise (px)
            'min_aspect': 1.8,             # bbox h/w below this => sitting/bending, don't trust height
            # keypoint gating / noise
            'ankle_conf_min': 0.5,
            'edge_margin': 5,
            'sigma_pix': 4.0,
            'sigma_joint': 0.01,
            'sigma_min': 0.05,
            'min_depression_deg': 2.0,
            'max_range': 6.0,
            # motion handling
            'sigma_time': 0.03,        # s; residual timing uncertainty -> covariance grows with joint rate
            'exposure_s': 0.03,        # s; motion blur of bbox edges = fy * tilt_rate * exposure
            'max_sigma_r': 1.0,        # drop ground-plane estimates whose radial sigma exceeds this (m)
            # online fit of tilt_offset and joint_lag from full-body detections
            'max_wait': 0.3,           # s; if the delayed joint reading hasn't arrived by then, use the newest one
            'calibrate': True,
            'calib_report_every': 100,
            'debug_image': False,
            # --- FITTED COVARIANCE MODEL (static grid test, held-out residuals) ---
            # Replaces the physical covariance above with, in polar form about pt_base_link (r = corrected range):
            #   sigma_r = sqrt((c_r r^n_r)^2 + s_r0^2 + (h r)^2 + mot_r^2)   h = sigma_person_height/person_height (ext only)
            #   sigma_t = sqrt((c_t r)^2 + s_t0^2 + mot_t^2)
            # mot_* = the node's own joint-rate / motion-blur terms, kept so a moving camera is still inflated.
            # The fit assumes the pan_offset above and the range correction below (r_corr = (r - b) / a).
            'use_fitted_cov': True,        # false = original physical covariance, no corrections, no gating
            'apply_corrections': True,     # range correction used when the model was fitted
            'gate_estimates': True,        # drop estimates outside the tested range / image bearing
            'max_image_bearing_deg': 35.0, # |atan((u-cx)/fx)|; the grid test covered up to the 30 deg node
            'ext_c_r': 0.01546, 'ext_n_r': 1.0, 'ext_s_r0': 0.0586,
            'ext_c_t': 0.01775, 'ext_s_t0': 0.0329,
            'ext_range_a': 0.9816, 'ext_range_b': 0.2255,
            'ext_r_min': 1.9, 'ext_r_max': 4.5,      # m, tested range (after correction)
            'gnd_c_r': 0.1082, 'gnd_n_r': 2.0, 'gnd_s_r0': 0.1826,
            'gnd_c_t': 0.0220, 'gnd_s_t0': 0.0279,
            'gnd_range_a': 1.0, 'gnd_range_b': 0.0,  # no range correction for gnd (error depends on bearing)
            'gnd_r_min': 1.0, 'gnd_r_max': 2.5,
            # --- LOGGING ---
            'log_enabled': False,
            'log_dir': '~/pt_logs',
            'marker_topic': '/eval/marker',   # std_msgs/String: 'start <node_id> <visit_id>' / 'stop'
        }
        for k, v in defaults.items():
            self.declare_parameter(k, v)
        for k in defaults:
            setattr(self, k, self.get_parameter(k).value)

        # Rectified-image intrinsics = projection matrix P (ugv_vision/config/camera_info.yaml)
        self.fx, self.cx = 196.89772, 342.88724
        self.fy, self.cy = 234.53159, 231.54267

        self.joint_buf = deque(maxlen=3000)      # (t, pan, tilt) raw readings
        self.pending = deque(maxlen=50)          # detections waiting for their delayed joint readings
        self.cal_samples = deque(maxlen=400)     # (t_image, true tilt implied by the image)
        self.cal_new = 0
        self.stats = dict(images=0, with_people=0, joint_msgs=0, finalized=0, timed_out=0,
                          dropped=0, no_joints=0, ext_ok=0, gnd_ok=0)
        self.last_img_t = None
        self.last_joint_t = None

        self.get_logger().info(f'Loading YOLO model from {self.model_path} ...')
        self.model = YOLO(self.model_path)
        self.get_logger().info('Model loaded.')

        # YOLO takes ~70 ms per frame. On a single-threaded executor with deep queues, that starves the joint
        # callback and lets frames/joint messages pile up (STATUS showed 0.6 s old frames, 1.6 s old joint data).
        # So: separate callback groups + MultiThreadedExecutor, a lock for shared state, and depth-1 image queue.
        self.lock = threading.RLock()
        self.img_group = MutuallyExclusiveCallbackGroup()
        self.joint_group = MutuallyExclusiveCallbackGroup()
        self.create_subscription(CompressedImage, self.image_topic, self.image_callback, 1,
                                 callback_group=self.img_group)
        self.create_subscription(JointState, '/ugv/joint_states', self.joint_state_callback, 10,
                                 callback_group=self.joint_group)

        self.detection_publisher = self.create_publisher(Detection2DArray, self.detections_topic, 10)
        self.pose_publisher = self.create_publisher(PoseWithCovarianceStamped, self.poses_topic, 10)
        self.ground_pose_publisher = self.create_publisher(
            PoseWithCovarianceStamped, self.poses_topic + '_ground', 10)
        self.debug_pub = self.create_publisher(CompressedImage, '/pantilt/debug/compressed', 1)

        self.create_timer(0.05, self.process_pending)   # flushes detections whose joint reading is late
        self.create_timer(2.0, self.status)

        self._init_logging(defaults)   # --- LOGGING ---

    # ------------------------------------------------------------------ logging (added)
    def _init_logging(self, defaults):
        self._label = ('idle', '', '')            # (state, node_id, visit_id); replaced atomically
        self._log_lock = threading.Lock()
        self._det_f = self._pose_f = None
        if not self.log_enabled:
            return
        d = os.path.expanduser(self.log_dir)
        os.makedirs(d, exist_ok=True)
        tag = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._det_f = open(os.path.join(d, f'pt_detections_{tag}.csv'), 'w', newline='')
        self._det_w = csv.writer(self._det_f)
        self._det_w.writerow(DET_COLS)
        self._pose_f = open(os.path.join(d, f'pt_poses_{tag}.csv'), 'w', newline='')
        self._pose_w = csv.writer(self._pose_f)
        self._pose_w.writerow(POSE_COLS)
        # snapshot of every setting, for the report's configuration table
        cfg = {k: getattr(self, k) for k in defaults}
        cfg.update(fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy)
        with open(os.path.join(d, f'pt_config_{tag}.json'), 'w') as f:
            json.dump(cfg, f, indent=2, default=str)
        self.marker_group = MutuallyExclusiveCallbackGroup()
        self.create_subscription(String, self.marker_topic, self.marker_callback, 10,
                                 callback_group=self.marker_group)
        self.get_logger().info(f'logging to {d} (tag {tag}); markers on {self.marker_topic}')

    def marker_callback(self, msg: String):
        """'start <node_id> <visit_id>' or 'stop'."""
        parts = msg.data.split()
        if len(parts) >= 3 and parts[0] == 'start':
            self._label = ('rec', parts[1], parts[2])
        elif parts and parts[0] == 'stop':
            self._label = ('idle', '', '')
        else:
            self.get_logger().warn(f"unrecognised marker '{msg.data}'")
            return
        self.get_logger().info(f'marker: {self._label[0]} {self._label[1]} {self._label[2]}')

    def log_detections(self, stamp, dets, infer_ms, w, h):
        """One row per detection, or one empty row (n_det=0) per processed image."""
        if self._det_f is None:
            return
        base = [self.get_clock().now().nanoseconds, stamp_to_ns(stamp), *self._label, len(dets)]
        tail = [f'{infer_ms:.1f}', w, h]
        with self._log_lock:
            if not dets:
                self._det_w.writerow(base + [''] * 12 + tail)
            for i, det in enumerate(dets):
                x1, y1, x2, y2 = det['box']
                if det['kxy'] is not None:
                    k = [_n(det['kxy'][LEFT_ANKLE][0]), _n(det['kxy'][LEFT_ANKLE][1]), _n(det['kconf'][LEFT_ANKLE]),
                         _n(det['kxy'][RIGHT_ANKLE][0]), _n(det['kxy'][RIGHT_ANKLE][1]), _n(det['kconf'][RIGHT_ANKLE])]
                else:
                    k = [''] * 6
                self._det_w.writerow(base + [i, _n(det['conf']), _n(x1), _n(y1), _n(x2), _n(y2)] + k + tail)
            self._det_f.flush()

    def log_pose(self, p, status, joints=None, ready=True, best_ext=None, best_gnd=None):
        """One row per pending detection frame, including ones that failed ('dropped', 'no_joints') or
        produced no estimate (ext_ok = gnd_ok = 0). Positions are as PUBLISHED (sign-flipped, frame pt_base_link),
        i.e. AFTER the range correction, and the covariance is the one published. ext_ok = 0 also covers estimates
        removed by the validity gates (use_fitted_cov / gate_estimates)."""
        if self._pose_f is None:
            return
        state, node, visit = p.get('lab', self._label)
        pan, tilt, pr, tr = joints if joints is not None else (None, None, None, None)

        def est(best):
            if best is None:
                return [0] + [''] * 8
            _, e, idx = best
            cov, extra = e[2], e[4]
            return [1, _n(-e[0]), _n(-e[1]), _n(cov[0, 0]), _n(cov[0, 1]), _n(cov[1, 1]),
                    _n(extra[0]), _n(extra[-1]), idx]

        row = [self.get_clock().now().nanoseconds, stamp_to_ns(p['stamp']), state, node, visit, status, int(ready),
               len(p['dets']),
               _n(math.degrees(pan)) if pan is not None else '', _n(math.degrees(tilt)) if tilt is not None else '',
               _n(math.degrees(pr)) if pr is not None else '', _n(math.degrees(tr)) if tr is not None else '']
        with self._log_lock:
            self._pose_w.writerow(row + est(best_ext) + est(best_gnd))
            self._pose_f.flush()

    def destroy_node(self):
        for f in (self._det_f, self._pose_f):
            if f is not None:
                f.close()
        super().destroy_node()

    # ------------------------------------------------------------------ joints
    def joint_state_callback(self, msg: JointState):
        try:
            pan = msg.position[msg.name.index(self.pan_joint)]
            tilt = msg.position[msg.name.index(self.tilt_joint)]
        except ValueError:
            self.get_logger().warn(
                f'Joints {self.pan_joint}/{self.tilt_joint} not in JointState {list(msg.name)}',
                throttle_duration_sec=5.0)
            return
        t = stamp_to_sec(msg.header.stamp)
        if t == 0.0:
            t = self.get_clock().now().nanoseconds * 1e-9
        with self.lock:
            self.joint_buf.append((t, self.pan_sign * pan + self.pan_offset, self.tilt_sign * tilt))
            self.stats['joint_msgs'] += 1
            self.last_joint_t = t
            self.process_pending()

    def joints_at(self, t, tol=0.2):
        """Raw pan/tilt at time t plus signed rates (rad/s, backward difference), or None."""
        if len(self.joint_buf) < 2:
            return None
        buf = np.asarray(self.joint_buf)
        if t < buf[0, 0] - tol or t > buf[-1, 0] + tol:
            return None
        pan = float(np.interp(t, buf[:, 0], buf[:, 1]))
        tilt = float(np.interp(t, buf[:, 0], buf[:, 2]))
        w = 0.15
        pan_rate = (pan - np.interp(t - w, buf[:, 0], buf[:, 1])) / w
        tilt_rate = (tilt - np.interp(t - w, buf[:, 0], buf[:, 2])) / w
        return pan, tilt, float(pan_rate), float(tilt_rate)

    # ---------------------------------------------------------------- geometry
    def camera_height(self, tilt_eff):
        return self.cam_base_height + self.cam_arm * math.sin(tilt_eff)

    def ray_local(self, u, v, tilt_eff):
        """Ray through pixel (u, v) in pan-frame axes (x fwd, y left, z up), camera tilted UP by tilt_eff."""
        xn = (u - self.cx) / self.fx
        yn = (v - self.cy) / self.fy
        return (math.cos(tilt_eff) + yn * math.sin(tilt_eff),
                -xn,
                math.sin(tilt_eff) - yn * math.cos(tilt_eff))

    def to_output(self, pan, tilt_eff, gx, gy):
        """Horizontal offset (gx, gy) from the camera, pan-frame -> output frame."""
        c, s = math.cos(pan), math.sin(pan)
        arm = self.cam_arm * math.cos(tilt_eff)
        return (self.pan_axis_x + arm * c + c * gx - s * gy,
                self.pan_axis_y + arm * s + s * gx + c * gy)

    @staticmethod
    def polar_cov(az, sig_r, sig_lat):
        R = np.array([[math.cos(az), -math.sin(az)], [math.sin(az), math.cos(az)]])
        return R @ np.diag([sig_r ** 2, sig_lat ** 2]) @ R.T

    def feet_pixel(self, xy, conf, w, h):
        m = self.edge_margin
        pts = []
        for k in (LEFT_ANKLE, RIGHT_ANKLE):
            u, v = xy[k]
            if conf[k] >= self.ankle_conf_min and m < u < w - m and m < v < h - m:
                pts.append((u, v))
        if not pts:
            return None
        return float(np.mean([p[0] for p in pts])), float(max(p[1] for p in pts))

    def ground_estimate(self, u, v, pan, tilt, pan_rate=0.0, tilt_rate=0.0):
        """Ankle pixel -> floor intersection. Sensitive to tilt error; covariance grows with joint rate.
        Returns (x, y, cov, info, (r, sig_r), aux) or None. aux = motion-only sigmas (mot_r, mot_t) and the
        pixel column u, used by apply_model()."""
        tilt = tilt + self.tilt_offset
        dx, dy, dz = self.ray_local(u, v, tilt)
        if dz >= 0.0:
            return None
        h = self.camera_height(tilt) - self.ankle_height
        lam = h / -dz
        gx, gy = lam * dx, lam * dy
        r = math.hypot(gx, gy)
        delta = math.atan2(-dz, math.hypot(dx, dy))
        if delta < math.radians(self.min_depression_deg) or r > self.max_range:
            return None
        x, y = self.to_output(pan, tilt, gx, gy)

        dv = v - self.cy
        sig_tilt = math.hypot(self.sigma_joint, self.sigma_time * abs(tilt_rate))
        sig_pan = math.hypot(self.sigma_joint, self.sigma_time * abs(pan_rate))
        sig_delta = math.hypot(self.sigma_pix * self.fy / (self.fy ** 2 + dv ** 2), sig_tilt)
        sig_r = max(h / math.sin(delta) ** 2 * sig_delta, self.sigma_min)
        if sig_r > self.max_sigma_r:
            return None
        sig_lat = max(r * math.hypot(self.sigma_pix / self.fx, sig_pan), self.sigma_min)
        cov = self.polar_cov(pan + math.atan2(gy, gx), sig_r, sig_lat)
        aux = {'mot_r': h / math.sin(delta) ** 2 * self.sigma_time * abs(tilt_rate),
               'mot_t': r * self.sigma_time * abs(pan_rate), 'u': u}
        return x, y, cov, f'ground r={r:.2f}m sig={sig_r:.2f} v={v:.0f} dep={math.degrees(delta):.1f}deg', (r, sig_r), aux

    def extent_estimate(self, u, v_top, v_bot, pan, tilt, pan_rate=0.0, tilt_rate=0.0):
        """Full-body range from the angle subtended by head-top and sole rays.
        That angle comes from two rows of the same image, so range does not depend on the joint readings:
        it is valid while the camera is moving. Joints enter only the bearing (pan) and, negligibly, the
        camera height. Returns (x, y, cov, info, (d, theta_bot, hc, sig_r), aux) or None. aux = motion-only
        sigmas (mot_r, mot_t) and the pixel column u, used by apply_model()."""
        tilt_eff = tilt + self.tilt_offset
        hc = self.camera_height(tilt_eff)
        top_h = self.person_height - hc
        if top_h <= 0.2:
            return None

        xn = (u - self.cx) / self.fx
        a = np.array([xn, (v_top - self.cy) / self.fy, 1.0]); a /= np.linalg.norm(a)
        b = np.array([xn, (v_bot - self.cy) / self.fy, 1.0]); b /= np.linalg.norm(b)
        phi = math.acos(float(np.clip(a @ b, -1.0, 1.0)))

        # solve atan(top_h/d) + atan(hc/d) = phi for the horizontal distance d (monotone decreasing in d)
        def f(d):
            return math.atan2(top_h, d) + math.atan2(hc, d) - phi
        lo, hi = 0.3, self.max_range
        if f(lo) < 0.0 or f(hi) > 0.0:
            return None
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if f(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        d = 0.5 * (lo + hi)

        # position: distance d along the azimuth of the ray through the box centre
        dx, dy, _ = self.ray_local(u, 0.5 * (v_top + v_bot), tilt_eff)
        az_local = math.atan2(dy, dx)
        x, y = self.to_output(pan, tilt_eff, d * math.cos(az_local), d * math.sin(az_local))

        dphi_dd = top_h / (d * d + top_h * top_h) + hc / (d * d + hc * hc)
        dphi_dH = d / (d * d + top_h * top_h)
        sig_box = math.hypot(self.sigma_box_pix, self.fy * abs(tilt_rate) * self.exposure_s)  # + motion blur
        sig_phi = math.sqrt(2.0) * sig_box / self.fy
        sig_d = math.hypot(sig_phi / dphi_dd, dphi_dH / dphi_dd * self.sigma_person_height)
        sig_r = max(sig_d, self.sigma_min)
        sig_pan = math.hypot(self.sigma_joint, self.sigma_time * abs(pan_rate))       # bearing depends on pan
        sig_lat = max(d * math.hypot(self.sigma_pix / self.fx, sig_pan), self.sigma_min)
        cov = self.polar_cov(pan + az_local, sig_r, sig_lat)
        aux = {'mot_r': math.sqrt(2.0) * abs(tilt_rate) * self.exposure_s / dphi_dd,
               'mot_t': d * self.sigma_time * abs(pan_rate), 'u': u}

        theta_bot = math.atan((v_bot - self.cy) / self.fy)
        return x, y, cov, f'extent d={d:.2f}m sig={sig_r:.2f}', (d, theta_bot, hc, sig_r), aux

    # ------------------------------------------------ fitted covariance model (static grid test)
    def apply_model(self, kind, est):
        """kind = 'ext' | 'gnd'. Replace the physical covariance of an estimate by the covariance function fitted
        on held-out residuals of the static grid test, after the range correction used in that fit, and drop
        estimates outside the tested envelope. Motion terms of the original model are added in quadrature.
        Returns a new estimate tuple, or None. With use_fitted_cov false the estimate is returned unchanged."""
        if est is None or not self.use_fitted_cov:
            return est
        x, y, _, info, extra, aux = est
        r = math.hypot(x, y)                         # range from pt_base_link (sign flip does not change it)
        if r < 1e-3:
            return None
        if self.apply_corrections:
            a = getattr(self, f'{kind}_range_a')
            b = getattr(self, f'{kind}_range_b')
            r_c = (r - b) / a
            if r_c <= 0.0:
                return None
            x, y, r = x * r_c / r, y * r_c / r, r_c
        if self.gate_estimates:
            if not (getattr(self, f'{kind}_r_min') <= r <= getattr(self, f'{kind}_r_max')):
                return None
            if abs(math.atan((aux['u'] - self.cx) / self.fx)) > math.radians(self.max_image_bearing_deg):
                return None
        c_r, n_r, s_r0 = (getattr(self, f'{kind}_{k}') for k in ('c_r', 'n_r', 's_r0'))
        c_t, s_t0 = getattr(self, f'{kind}_c_t'), getattr(self, f'{kind}_s_t0')
        h = self.sigma_person_height / self.person_height if kind == 'ext' else 0.0
        sig_r = math.sqrt((c_r * r ** n_r) ** 2 + s_r0 ** 2 + (h * r) ** 2 + aux['mot_r'] ** 2)
        sig_t = math.sqrt((c_t * r) ** 2 + s_t0 ** 2 + aux['mot_t'] ** 2)
        cov = self.polar_cov(math.atan2(y, x), sig_r, sig_t)
        return x, y, cov, info + ' [fit]', tuple(extra[:-1]) + (sig_r,), aux

    # --------------------------------------------------- tilt offset / lag calibration
    def calib_add(self, t_img, true_tilt):
        """true_tilt = camera pitch implied by the image (sole ray minus range-derived depression)."""
        self.cal_samples.append((t_img, true_tilt))
        self.cal_new += 1
        if self.cal_new >= self.calib_report_every:
            self.cal_new = 0
            self.calib_report()

    def calib_report(self):
        """Find joint_lag and tilt_offset so that image-implied tilt(t) = reading(t + lag) + offset."""
        if len(self.cal_samples) < 50 or len(self.joint_buf) < 20:
            return
        js = np.asarray(self.joint_buf)
        ts = np.array([s[0] for s in self.cal_samples])
        ys = np.array([s[1] for s in self.cal_samples])
        ok = (ts >= js[0, 0]) & (ts + 0.6 <= js[-1, 0])
        if ok.sum() < 50:
            return
        ts, ys = ts[ok], ys[ok]
        motion = math.degrees(float(np.std(np.interp(ts, js[:, 0], js[:, 2]))))
        if motion < 2.0:   # camera (nearly) static: only the offset is observable
            resid = ys - np.interp(ts + self.joint_lag, js[:, 0], js[:, 2])
            self.get_logger().info(
                f'CALIB n={len(ts)} (tilt moved only {motion:.1f} deg): tilt_offset={resid.mean():.4f} rad '
                f'({math.degrees(resid.mean()):.2f} deg) at joint_lag={self.joint_lag:.3f}; slew the camera to fit lag')
            return
        best = None
        for lag in np.arange(0.0, 0.6, 0.01):
            resid = ys - np.interp(ts + lag, js[:, 0], js[:, 2])
            v = float(np.var(resid))
            if best is None or v < best[0]:
                best = (v, lag, float(resid.mean()))
        v, lag, off = best
        self.get_logger().info(
            f'CALIB n={len(ts)}: joint_lag={lag:.2f} s, tilt_offset={off:.4f} rad ({math.degrees(off):.2f} deg), '
            f'resid std {math.degrees(math.sqrt(v)):.2f} deg (tilt moved {motion:.1f} deg)')

    # ---------------------------------------------------------------- outputs
    def make_pose_msg(self, stamp, x, y, cov):
        out = PoseWithCovarianceStamped()
        out.header.stamp = stamp
        out.header.frame_id = self.output_frame
        out.pose.pose.position.x = -x
        out.pose.pose.position.y = -y
        out.pose.pose.position.z = 0.0
        out.pose.pose.orientation.w = 1.0
        C = np.diag([0.0, 0.0, 1e6, 1e6, 1e6, 1e6])
        C[0:2, 0:2] = cov
        out.pose.covariance = C.flatten().tolist()
        return out

    def publish_poses(self, stamp, best_ext, best_gnd):
        self.stats['ext_ok'] += best_ext is not None
        self.stats['gnd_ok'] += best_gnd is not None
        if best_ext is not None:
            self.pose_publisher.publish(self.make_pose_msg(stamp, *best_ext[1][:3]))
        if best_gnd is not None:
            self.ground_pose_publisher.publish(self.make_pose_msg(stamp, *best_gnd[1][:3]))

    # ---------------------------------------------------------------- pipeline
    def enqueue(self, p):
        with self.lock:
            self.stats['with_people'] += 1
            self.pending.append(p)
            self.process_pending()

    def process_pending(self):
        with self.lock:
            self._process_pending()

    def _process_pending(self):
        """Finish detections once the joint readings for (image time + joint_lag) have arrived.
        If they are late (sparse or lagging joint stream) use the newest reading after max_wait."""
        if not self.pending:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        t_newest = self.joint_buf[-1][0] if self.joint_buf else None
        while self.pending:
            p = self.pending[0]
            waited = now - p['arrived']
            ready = t_newest is not None and p['t'] + self.joint_lag <= t_newest
            if not ready and waited < self.max_wait:
                break
            self.pending.popleft()
            if t_newest is None or waited > 1.5:
                self.stats['dropped'] += 1
                self.log_pose(p, 'dropped')   # --- LOGGING ---
                continue
            if not ready:
                self.stats['timed_out'] += 1
            self.finalize(p, ready)

    def status(self):
        with self.lock:
            self._status()

    def _status(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        s = self.stats
        ia = f'{now - self.last_img_t:.1f}s' if self.last_img_t is not None else 'never'
        ja = f'{now - self.last_joint_t:.1f}s' if self.last_joint_t is not None else 'never'
        hint = ''
        for age in (self.last_img_t, self.last_joint_t):
            if age is not None and abs(now - age) > 5.0:
                hint = '  <-- stamp far from node clock: clock-domain mismatch?'
        #self.get_logger().info(
        #    f"STATUS images={s['images']} with_people={s['with_people']} joint_msgs={s['joint_msgs']} "
        #    f"pending={len(self.pending)} finalized={s['finalized']} (timed_out={s['timed_out']} "
        #    f"dropped={s['dropped']} no_joints={s['no_joints']}) ext_ok={s['ext_ok']} gnd_ok={s['gnd_ok']} "
        #    f"| image stamp age={ia}, joint stamp age={ja}{hint}")

    def finalize(self, p, ready=True):
        joints = self.joints_at(p['t'] + self.joint_lag, tol=0.2 if ready else 0.6)
        if joints is None:
            self.stats['no_joints'] += 1
            self.log_pose(p, 'no_joints', None, ready)   # --- LOGGING ---
            return
        self.stats['finalized'] += 1
        pan, tilt, pan_rate, tilt_rate = joints
        m, w_img, h_img = self.edge_margin, p['w'], p['h']
        best_ext = best_gnd = None

        for i, det in enumerate(p['dets']):
            x1, y1, x2, y2 = det['box']
            u_c = (x1 + x2) / 2.0
            score = abs(u_c - self.cx)   # crude association: nearest the IBVS column

            ext_raw = gnd_raw = None
            if y1 > m and y2 < h_img - m and (y2 - y1) >= self.min_aspect * (x2 - x1):
                ext_raw = self.extent_estimate(u_c, y1, y2, pan, tilt, pan_rate, tilt_rate)
            if det['kxy'] is not None:
                feet = self.feet_pixel(det['kxy'], det['kconf'], w_img, h_img)
                if feet is not None:
                    gnd_raw = self.ground_estimate(feet[0], feet[1], pan, tilt, pan_rate, tilt_rate)

            # tilt/lag calibration uses the physical (uncorrected) extent estimate
            if ext_raw is not None and self.calibrate and abs(u_c - self.cx) < 80:
                d, th_bot, hc = ext_raw[4][:3]
                self.calib_add(p['t'], th_bot - math.atan2(hc, d))

            # fitted covariance + range correction + validity gates (no-op if use_fitted_cov is false)
            ext = self.apply_model('ext', ext_raw)
            gnd = self.apply_model('gnd', gnd_raw)

            if ext is not None and (best_ext is None or score < best_ext[0]):
                best_ext = (score, ext, i)
            if gnd is not None and (best_gnd is None or score < best_gnd[0]):
                best_gnd = (score, gnd, i)

            def fmt(e):
                return f'{e[3]} -> ({e[0]:.2f},{e[1]:.2f})' if e is not None else 'none'
            self.get_logger().info(
                f'pan={math.degrees(pan):.1f} tilt_raw={math.degrees(tilt):.1f} '
                f'rate(p,t)=({math.degrees(pan_rate):.0f},{math.degrees(tilt_rate):.0f})deg/s | '
                f'bbox y=[{y1:.0f},{y2:.0f}] | ext: {fmt(ext)} | gnd: {fmt(gnd)}',
                throttle_duration_sec=0.5)

        self.publish_poses(p['stamp'], best_ext, best_gnd)
        self.log_pose(p, 'ok' if ready else 'timed_out', joints, ready, best_ext, best_gnd)   # --- LOGGING ---

    def image_callback(self, msg: CompressedImage):
        frame_bgr = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            self.get_logger().warn('Failed to decode incoming CompressedImage.')
            return

        t_inf = time.perf_counter()   # --- LOGGING ---
        results = self.model.predict(
            source=frame_bgr, conf=self.confidence_threshold,
            classes=[PERSON_CLASS_ID], device=self.device, verbose=False)
        infer_ms = (time.perf_counter() - t_inf) * 1e3   # --- LOGGING ---

        t_img = stamp_to_sec(msg.header.stamp)
        with self.lock:
            self.last_img_t = t_img
            self.stats['images'] += 1
        detection_array = Detection2DArray()
        detection_array.header = msg.header
        dets = []
        h_img, w_img = frame_bgr.shape[:2]

        if results:
            r = results[0]
            have_kpts = r.keypoints is not None and r.keypoints.conf is not None
            if have_kpts:
                kxy = r.keypoints.xy.cpu().numpy()
                kconf = r.keypoints.conf.cpu().numpy()

            for i, box in enumerate(r.boxes):
                if int(box.cls[0]) != PERSON_CLASS_ID:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detection = Detection2D()
                detection.header = msg.header
                bbox = BoundingBox2D()
                bbox.center.position.x = (x1 + x2) / 2.0
                bbox.center.position.y = (y1 + y2) / 2.0
                bbox.center.theta = 0.0
                bbox.size_x = x2 - x1
                bbox.size_y = y2 - y1
                detection.bbox = bbox
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = str(PERSON_CLASS_ID)
                hyp.hypothesis.score = float(box.conf[0])
                detection.results.append(hyp)
                detection_array.detections.append(detection)

                dets.append({'box': (x1, y1, x2, y2),
                             'conf': float(box.conf[0]),   # --- LOGGING ---
                             'kxy': kxy[i] if have_kpts else None,
                             'kconf': kconf[i] if have_kpts else None})

                if self.debug_image and have_kpts:
                    for k in (LEFT_ANKLE, RIGHT_ANKLE):
                        ku, kv = int(kxy[i][k][0]), int(kxy[i][k][1])
                        cv2.circle(frame_bgr, (ku, kv), 4, (0, 0, 255), -1)
                        cv2.putText(frame_bgr, f'{kconf[i][k]:.2f}', (ku + 6, kv),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # --- LOGGING: every processed image, including those with no person (needed for recall)
        self.log_detections(msg.header.stamp, dets, infer_ms, w_img, h_img)

        # IBVS consumes this immediately; the pose estimate follows once the joint readings catch up.
        self.detection_publisher.publish(detection_array)

        if dets:
            self.enqueue({'t': t_img, 'stamp': msg.header.stamp, 'w': w_img, 'h': h_img, 'dets': dets,
                          'arrived': self.get_clock().now().nanoseconds * 1e-9,
                          'lab': self._label})   # --- LOGGING: label at image time

        if self.debug_image:
            with self.lock:
                joints = self.joints_at(t_img + self.joint_lag)
            if joints is not None:
                v_h = self.cy - self.fy * math.tan(joints[1] + self.tilt_offset)   # predicted horizon row
                if 0 <= v_h < h_img:
                    cv2.line(frame_bgr, (0, int(v_h)), (w_img, int(v_h)), (0, 255, 255), 1)
            ok, buf = cv2.imencode('.jpg', frame_bgr)
            if ok:
                dbg = CompressedImage()
                dbg.header = msg.header
                dbg.format = 'jpeg'
                dbg.data = buf.tobytes()
                self.debug_pub.publish(dbg)


def main(args=None):
    rclpy.init(args=args)
    node = HumanDetector()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()