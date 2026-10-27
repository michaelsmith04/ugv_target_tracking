#!/usr/bin/env python3
import csv
import json
import math
import os
from collections import deque
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState, CameraInfo
from nav_msgs.msg import Odometry
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String
from geometry_msgs.msg import Vector3

JOINT_NAMES = ['pt_base_link_to_pt_link1', 'pt_link1_to_pt_link2']
STEPS_PER_RAD = 4096.0 / (2.0 * math.pi)   # ST3125: 4096 steps / rev

# --- LOGGING: column definitions ---------------------------------------------------------------------------
EV_COLS = ['recv_ns', 'stamp_ns', 'label', 'status', 'n_det', 'score', 'cx', 'cy', 'ex_px', 'ey_px', 'age_s',
           'err_pan', 'err_tilt', 'pan_cap', 'tilt_cap', 'yaw_cap', 'body_yaw',
           'pan_goal_before', 'tilt_goal_before', 'rem_pan', 'rem_tilt', 'rem_pan_db', 'rem_tilt_db',
           'pan_goal_after', 'tilt_goal_after', 'applied', 'reason']
TICK_COLS = ['recv_ns', 'dt', 'label', 'mode', 'pan_goal', 'tilt_goal', 'pan_est', 'tilt_est',
             'pan_spd', 'tilt_spd', 'body_rate', 'body_yaw', 'target_fresh', 'odom_fresh', 'sweeping',
             'odom_x', 'odom_y', 'odom_z', 'ukf_pan', 'ukf_tilt', 'test_hold']


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def wrap_pi(a):
    """Wrap an angle to [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _n(v):
    return '' if v is None else f'{float(v):.6g}'


class PanTilt(Node):
    """
    Open-loop pan/tilt tracker (no servo feedback available).

    Conventions (same as your original code):
      +pan  = camera turns toward image-right (clockwise seen from above)
      +tilt = camera turns up
    """

    def __init__(self):
        super().__init__('pan_tilt')

        # ---------------- tunables ----------------
        # Pan wrap-around. The servo range maps -pi..+pi onto 0..4094 steps, so the seam is
        # the two ends of the range: crossing it makes the servo SWEEP the whole way back
        # through the front (~2*pi rad, ~2 s). Set PAN_WRAP=False to clamp at +-PAN_LIMIT instead.
        self.PAN_WRAP = True
        self.PAN_LIMIT = math.pi            # only used when PAN_WRAP is False
        self.PAN_EDGE = math.pi - 0.002     # clamp used to sit at the seam without wrapping
        self.SWEEP_DONE_RAD = 0.15          # sweep is over when est is this close to the goal
        self.SEAM_COOLDOWN_SEC = 2.0        # min time between seam crossings (anti-thrash)
        self.TILT_LIMIT = math.pi / 2       # placeholder
        self.MAX_PAN_RATE = 2.5             # rad/s, what the estimate assumes. MEASURE.
        self.MAX_TILT_RATE = 2.5            # rad/s. MEASURE.
        self.MAX_LEAD = 1.0                 # rad the goal may run ahead of the estimate
        self.HOMING_SEC = 3.0

        # IBVS: fraction of the (latency-compensated) angular error corrected per detection.
        # Start ~0.5, raise until it overshoots, back off 20%.
        self.K_PAN = 0.5
        self.K_TILT = 0.3
        self.PIXEL_DEADBAND = 8             # applied to the *predicted residual*
        self.LATENCY_FALLBACK_SEC = 0.12    # image->command delay if the stamp is unusable
        self.LOST_HOLD_SEC = 2.0          # ride out single-frame dropouts before UKF takes over
        self.IBVS_ACTIVE_SEC = 0.3          # label only: IBVS if a detection was this recent, else HOLD

        # Rover yaw-rate feedforward. Camera pans against body rotation to hold heading.
        # Sign: with +pan = clockwise and ROS yaw CCW-positive, +1 is expected. Flip if wrong.
        self.ROVER_ODOM_TOPIC = 'odom'      # CHECK: any nav_msgs/Odometry with the rover twist
        self.FF_GAIN = 0.2                  # 0 disables
        self.BODY_RATE_TIMEOUT_SEC = 0.5

        # Search scan
        self.SEARCH_RATE = 0.6              # rad/s, ~23 deg/s. Slow enough for YOLO, no blur
        self.SEARCH_PAN_LIMIT = math.pi - 0.4   # scan between +-this. Stays clear of the seam
        self.SEARCH_TILT = 0.26              # rad, fixed tilt while scanning
        self.SEARCH_LEAD_SEC = 0.25         # goal runs this far ahead of the estimate

        # UKF fallback
        self.ENABLE_UKF_FALLBACK = False     # False = hold when there is no detection
        self.ODOM_TIMEOUT_SEC = 1.0
        # Axis convention of target/odometry_filtered position:
        #   'rep103'  : x forward, y left,  z up       (standard ROS base frame; original formula)
        #   'optical' : x right,   y up,    z forward  (OAK-style camera convention)
        self.UKF_AXES = 'rep103'
        self.UKF_PAN_SIGN = 1.0             # flip if the camera swings the wrong way
        self.UKF_TILT_SIGN = 1.0

        # Servo speed command (steps/s). Firmware clamps SX/SY to 1..2500 (~3.8 rad/s).
        self.SPD_MIN = 1200.0
        self.SPD_MAX = 2500.0
        self.SPEED_HORIZON_SEC = 0.1        # servo aims to close the goal-est gap in this time

        self.CONTROL_PERIOD_SEC = 0.02      # 50 Hz

        # --- LOGGING / TEST HOOKS (added) ---
        self.LOG_ENABLED = False
        self.LOG_DIR = '~/pt_act_logs'
        self.LOG_TICK_STRIDE = 1            # 1 = log every 50 Hz control tick
        self.MARKER_TOPIC = '/eval/marker'  # std_msgs/String; any text, logged verbatim and used as the label
        self.TEST_CMD_TOPIC = 'pantilt/test_cmd'   # geometry_msgs/Vector3, see test_cmd_callback

        # ---------------- state ----------------
        self.odom_x = self.odom_y = self.odom_z = 0.0
        self.odom_s = -1e9
        self.mode = 'HOME'
        self.search_dir = 1.0
        self.last_ex = 0.0

        self.pan_est = 0.0      # believed actual camera angle
        self.tilt_est = 0.0
        self.pan_goal = 0.0     # last commanded goal (persistent)
        self.tilt_goal = 0.0
        self.pan_sweeping = False   # True while the servo is travelling across the seam
        self.sweep_end_s = -1e9

        self.body_rate = 0.0    # rover yaw rate (rad/s)
        self.body_rate_s = -1e9
        self.body_yaw = 0.0     # integrated rover yaw (only differences are used)

        self.image_width = 640
        self.image_height = 480
        self.fx = self.fy = 457.0   # ~70 deg HFOV at 640 px; overwritten by camera_info
        self.last_target_s = -1e9

        # (t, pan_est, tilt_est, body_yaw) history for latency compensation
        self.hist = deque(maxlen=150)

        self.test_hold = False  # --- TEST: True = IBVS/UKF suspended, goals set by test_cmd only
        self.label = ''         # --- LOGGING: last marker text ('' after 'stop')
        self._tick_n = 0

        self.start_s = self.now_s()
        self.last_loop_s = self.start_s

        # ---------------- ROS I/O ----------------
        self.create_subscription(Odometry, 'target/odometry_filtered', self.odometry_callback, 10)
        self.create_subscription(Detection2DArray, 'pantilt/detections', self.pt_detection_callback, 10)
        self.create_subscription(CameraInfo, 'pantilt/camera_info', self.camera_info_callback, 10)
        self.create_subscription(Odometry, self.ROVER_ODOM_TOPIC, self.rover_odom_callback, 10)

        self.joint_pub = self.create_publisher(JointState, 'ugv/pt_cmd', 10)
        self.estimate_pub = self.create_publisher(JointState, 'pantilt/joint_estimate', 10)

        self.create_timer(self.CONTROL_PERIOD_SEC, self.control_loop)

        self._init_logging()   # --- LOGGING ---

    # ---------------- logging and test hooks (added) ----------------

    def _init_logging(self):
        self._ev_f = self._tick_f = self._mk_f = None
        if not self.LOG_ENABLED:
            return
        d = os.path.expanduser(self.LOG_DIR)
        os.makedirs(d, exist_ok=True)
        tag = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._ev_f = open(os.path.join(d, f'ibvs_events_{tag}.csv'), 'w', newline='')
        self._ev_w = csv.writer(self._ev_f)
        self._ev_w.writerow(EV_COLS)
        self._tick_f = open(os.path.join(d, f'ctrl_ticks_{tag}.csv'), 'w', newline='')
        self._tick_w = csv.writer(self._tick_f)
        self._tick_w.writerow(TICK_COLS)
        self._mk_f = open(os.path.join(d, f'markers_{tag}.csv'), 'w', newline='')
        self._mk_w = csv.writer(self._mk_f)
        self._mk_w.writerow(['recv_ns', 'kind', 'text'])
        cfg = {k: v for k, v in vars(self).items() if k.isupper()}
        cfg.update(fx=self.fx, fy=self.fy, image_width=self.image_width, image_height=self.image_height)
        with open(os.path.join(d, f'pt_act_config_{tag}.json'), 'w') as f:
            json.dump(cfg, f, indent=2, default=str)
        self.create_subscription(String, self.MARKER_TOPIC, self.marker_callback, 10)
        self.create_subscription(Vector3, self.TEST_CMD_TOPIC, self.test_cmd_callback, 10)
        self.get_logger().info(f'logging to {d} (tag {tag}); markers on {self.MARKER_TOPIC}; '
                               f'test commands on {self.TEST_CMD_TOPIC}')

    def _mk(self, kind, text):
        if self._mk_f is not None:
            self._mk_w.writerow([self.get_clock().now().nanoseconds, kind, text])
            self._mk_f.flush()

    def marker_callback(self, msg: String):
        """Any text is logged and becomes the label until the next marker; 'stop' clears it."""
        self._mk('marker', msg.data)
        self.label = '' if msg.data.strip() == 'stop' else msg.data.strip()

    def test_cmd_callback(self, msg: Vector3):
        """Test hook. x = pan [rad], y = tilt [rad], z = mode:
             0 : relative KICK, goals += (x, y); IBVS stays active and recovers (step response).
             1 : ABSOLUTE open-loop goal (x, y); IBVS and the UKF fallback are suspended (actuator test).
             2 : release, resume normal operation.
        Note MAX_LEAD (1 rad) still limits how far the goal may run ahead of the estimate."""
        mode = int(round(msg.z))
        if mode == 0:
            self.pan_goal += msg.x
            self.tilt_goal += msg.y
        elif mode == 1:
            self.test_hold = True
            self.pan_goal, self.tilt_goal = msg.x, msg.y
        elif mode == 2:
            self.test_hold = False
        self._mk('test_cmd', f'mode={mode} pan={msg.x:.5f} tilt={msg.y:.5f}')

    def log_ibvs(self, now_s, stamp_ns, status, n_det, **kw):
        if self._ev_f is None:
            return
        row = dict(recv_ns=int(now_s * 1e9), stamp_ns=stamp_ns, label=self.label, status=status, n_det=n_det)
        row.update(kw)
        self._ev_w.writerow([row.get(c, '') if isinstance(row.get(c, ''), (str, int)) else _n(row.get(c))
                             for c in EV_COLS])
        self._ev_f.flush()

    def log_tick(self, now_s, dt, mode, pan_spd, tilt_spd, target_fresh, odom_fresh):
        if self._tick_f is None:
            return
        self._tick_n += 1
        if self._tick_n % self.LOG_TICK_STRIDE:
            return
        up, ut = self.ukf_angles()
        self._tick_w.writerow([int(now_s * 1e9), _n(dt), self.label, mode, _n(self.pan_goal), _n(self.tilt_goal),
                               _n(self.pan_est), _n(self.tilt_est), _n(pan_spd), _n(tilt_spd),
                               _n(self.body_rate), _n(self.body_yaw), int(target_fresh), int(odom_fresh),
                               int(self.pan_sweeping), _n(self.odom_x), _n(self.odom_y), _n(self.odom_z),
                               _n(up), _n(ut), int(self.test_hold)])
        if self._tick_n % 50 == 0:
            self._tick_f.flush()

    def destroy_node(self):
        for f in (self._ev_f, self._tick_f, self._mk_f):
            if f is not None:
                f.close()
        super().destroy_node()

    # ---------------- helpers ----------------
    def search_step(self):
        """Triangle-wave scan. The goal runs a short lead ahead of the estimate, so the
        servo moves continuously at SEARCH_RATE instead of stop-start. Returns pan speed (steps/s)."""
        lim = self.SEARCH_PAN_LIMIT
        if self.pan_est >= lim - 0.02:
            self.search_dir = -1.0
        elif self.pan_est <= -lim + 0.02:
            self.search_dir = 1.0
        lead = self.SEARCH_RATE * self.SEARCH_LEAD_SEC
        self.pan_goal = clamp(self.pan_est + self.search_dir * lead, -lim, lim)
        self.tilt_goal = self.SEARCH_TILT
        return self.SEARCH_RATE * STEPS_PER_RAD


    def now_s(self):
        return self.get_clock().now().nanoseconds / 1e9

    def current_body_rate(self, now_s):
        if now_s - self.body_rate_s > self.BODY_RATE_TIMEOUT_SEC:
            return 0.0
        return self.body_rate

    def ukf_angles(self):
        x, y, z = self.odom_x, self.odom_y, self.odom_z
        if self.UKF_AXES == 'optical':
            fwd, right, up = z, x, y
        else:                                   # rep103
            fwd, right, up = x, -y, z
        pan = math.atan2(right, fwd)            # +pan = clockwise = target on the right
        tilt = math.atan2(up, math.hypot(fwd, right))
        return self.UKF_PAN_SIGN * pan, self.UKF_TILT_SIGN * tilt

    def history_at(self, t):
        if not self.hist:
            return self.pan_est, self.tilt_est, self.body_yaw
        h = min(self.hist, key=lambda s: abs(s[0] - t))
        return h[1], h[2], h[3]

    def limit_goals(self):
        # ---- pan ----
        if self.PAN_WRAP:
            goal = wrap_pi(self.pan_goal)
            if self.pan_sweeping:
                # Mid-sweep the goal is legitimately far from the estimate: no lead clamp.
                if abs(goal - self.pan_est) < self.SWEEP_DONE_RAD:
                    self.pan_sweeping = False
                    self.sweep_end_s = self.now_s()
            else:
                # Lead clamp in heading space, so a step across the seam is not blocked
                d = clamp(wrap_pi(goal - self.pan_est), -self.MAX_LEAD, self.MAX_LEAD)
                goal = wrap_pi(self.pan_est + d)
                if abs(goal - self.pan_est) > math.pi:          # this goal crosses the seam
                    if self.now_s() - self.sweep_end_s >= self.SEAM_COOLDOWN_SEC:
                        self.pan_sweeping = True
                    else:                                       # too soon: stay at the end
                        goal = clamp(self.pan_est + d, -self.PAN_EDGE, self.PAN_EDGE)
            self.pan_goal = goal
        else:
            self.pan_goal = clamp(self.pan_goal, -self.PAN_LIMIT, self.PAN_LIMIT)
            self.pan_goal = clamp(self.pan_goal, self.pan_est - self.MAX_LEAD,
                                  self.pan_est + self.MAX_LEAD)

        # ---- tilt ----
        self.tilt_goal = clamp(self.tilt_goal, -self.TILT_LIMIT, self.TILT_LIMIT)
        self.tilt_goal = clamp(self.tilt_goal, self.tilt_est - self.MAX_LEAD,
                               self.tilt_est + self.MAX_LEAD)

    def axis_speed(self, goal, est, ff_rate):
        rate = abs(goal - est) / self.SPEED_HORIZON_SEC + abs(ff_rate)
        return clamp(rate * STEPS_PER_RAD, self.SPD_MIN, self.SPD_MAX)

    def speeds(self, ff_rate):
        return (self.axis_speed(self.pan_goal, self.pan_est, ff_rate),
                self.axis_speed(self.tilt_goal, self.tilt_est, 0.0))

    @staticmethod
    def slew(target, est, max_rate, dt):
        step = max_rate * dt
        return est + clamp(target - est, -step, step)

    def advance_estimate(self, pan_spd, tilt_spd, dt):
        # Estimate moves at the commanded speed, capped by the assumed servo limit.
        pan_rate = min(self.MAX_PAN_RATE, pan_spd / STEPS_PER_RAD)
        tilt_rate = min(self.MAX_TILT_RATE, tilt_spd / STEPS_PER_RAD)
        self.pan_est = self.slew(self.pan_goal, self.pan_est, pan_rate, dt)
        self.tilt_est = self.slew(self.tilt_goal, self.tilt_est, tilt_rate, dt)

    # ---------------- subscriptions ----------------

    def camera_info_callback(self, msg):
        self.image_width = msg.width
        self.image_height = msg.height
        if msg.k[0] > 0.0 and msg.k[4] > 0.0:
            self.fx, self.fy = msg.k[0], msg.k[4]

    def odometry_callback(self, msg):
        self.odom_x = msg.pose.pose.position.x  # axes meaning set by UKF_AXES
        self.odom_y = msg.pose.pose.position.y
        self.odom_z = msg.pose.pose.position.z
        self.odom_s = self.now_s()

    def rover_odom_callback(self, msg):
        self.body_rate = msg.twist.twist.angular.z
        self.body_rate_s = self.now_s()

    def pt_detection_callback(self, msg):
        now_s = self.now_s()
        stamp_ns = msg.header.stamp.sec * 10 ** 9 + msg.header.stamp.nanosec
        # Empty messages are ignored: LOST_HOLD_SEC decides when the target is "lost".
        if not msg.detections:
            self.log_ibvs(now_s, stamp_ns, 'empty', 0)
            return
        if now_s - self.start_s < self.HOMING_SEC:
            self.log_ibvs(now_s, stamp_ns, 'homing', len(msg.detections))
            return

        best = max(
            msg.detections,
            key=lambda d: d.results[0].hypothesis.score if d.results else 0.0,
        )
        cx = best.bbox.center.position.x
        cy = best.bbox.center.position.y

        # When was the image captured? Use the header stamp if it is sane.
        age = now_s - Time.from_msg(msg.header.stamp).nanoseconds / 1e9
        if not (0.0 <= age <= 0.5):
            age = self.LATENCY_FALLBACK_SEC

        self.last_target_s = now_s
        self.apply_ibvs(cx, cy, now_s - age,
                        ctx=dict(now_s=now_s, stamp_ns=stamp_ns, n_det=len(msg.detections), age=age,
                                 score=best.results[0].hypothesis.score if best.results else None))

    # ---------------- IBVS (runs once per NEW detection) ----------------

    def apply_ibvs(self, cx, cy, t_cap, ctx=None):
        ex = cx - self.image_width / 2.0
        self.last_ex = ex
        ey = cy - self.image_height / 2.0

        # pixel error -> angular error (image y is down, +tilt is up)
        err_pan = math.atan2(ex, self.fx)
        err_tilt = -math.atan2(ey, self.fy)

        # Where was the camera / rover when this frame was taken?
        pan_cap, tilt_cap, yaw_cap = self.history_at(t_cap)

        # Remaining error = measured error, minus motion already commanded since capture
        # (so we never correct the same error twice), plus the rover's own rotation since capture.
        rem_pan = (err_pan - wrap_pi(self.pan_goal - pan_cap)
                   + self.FF_GAIN * (self.body_yaw - yaw_cap))
        rem_tilt = err_tilt - (self.tilt_goal - tilt_cap)
        rem_pan0, rem_tilt0 = rem_pan, rem_tilt
        pan_before, tilt_before = self.pan_goal, self.tilt_goal

        if abs(rem_pan) < math.atan2(self.PIXEL_DEADBAND, self.fx):
            rem_pan = 0.0
        if abs(rem_tilt) < math.atan2(self.PIXEL_DEADBAND, self.fy):
            rem_tilt = 0.0

        reason = ''
        if self.test_hold:              # --- TEST: log the pixel error but do not act on it
            reason = 'test_hold'
        else:
            if not self.pan_sweeping:       # mid-sweep the image says nothing useful about pan
                self.pan_goal += self.K_PAN * rem_pan
            else:
                reason = 'sweeping'
            self.tilt_goal += self.K_TILT * rem_tilt
            self.limit_goals()

            # Send immediately instead of waiting for the next timer tick
            ff_rate = self.FF_GAIN * self.current_body_rate(self.now_s())
            pan_spd, tilt_spd = self.speeds(ff_rate)
            self.publish_command(pan_spd, tilt_spd, 'IBVS')

        if ctx is not None:   # --- LOGGING
            self.log_ibvs(ctx['now_s'], ctx['stamp_ns'], 'ok', ctx['n_det'], score=ctx['score'], cx=cx, cy=cy,
                          ex_px=ex, ey_px=ey, age_s=ctx['age'], err_pan=err_pan, err_tilt=err_tilt,
                          pan_cap=pan_cap, tilt_cap=tilt_cap, yaw_cap=yaw_cap, body_yaw=self.body_yaw,
                          pan_goal_before=pan_before, tilt_goal_before=tilt_before,
                          rem_pan=rem_pan0, rem_tilt=rem_tilt0, rem_pan_db=rem_pan, rem_tilt_db=rem_tilt,
                          pan_goal_after=self.pan_goal, tilt_goal_after=self.tilt_goal,
                          applied=int(not self.test_hold), reason=reason)

    # ---------------- control loop (50 Hz) ----------------

    def control_loop(self):
        now_s = self.now_s()
        dt = clamp(now_s - self.last_loop_s, 0.0, 0.2)
        self.last_loop_s = now_s

        body_rate = self.current_body_rate(now_s)
        self.body_yaw += body_rate * dt
        self.hist.append((now_s, self.pan_est, self.tilt_est, self.body_yaw))

        # Homing: hold (0, 0) until the camera has certainly arrived there
        if now_s - self.start_s < self.HOMING_SEC:
            self.pan_goal = self.tilt_goal = 0.0
            self.publish_command(0.0, 0.0, 'HOME')
            self.log_tick(now_s, dt, 'HOME', 0.0, 0.0, False, False)
            return

        ff_rate = self.FF_GAIN * body_rate
        since = now_s - self.last_target_s
        target_fresh = since <= self.LOST_HOLD_SEC
        odom_fresh = (now_s - self.odom_s) <= self.ODOM_TIMEOUT_SEC   # logging only

        search_spd = None
        if self.test_hold:                      # --- TEST: goals come from test_cmd only
            mode = 'TEST'
            ff_rate = 0.0
        elif target_fresh:
            mode = 'IBVS' if since <= self.IBVS_ACTIVE_SEC else 'HOLD'
            if not self.pan_sweeping:
                self.pan_goal += ff_rate * dt   # hold heading against rover rotation
        else:
            mode = 'SEARCH'
            ff_rate = 0.0
            if self.mode != 'SEARCH':           # just entered: start toward the side it left on
                self.search_dir = 1.0 if self.last_ex >= 0 else -1.0
                self.get_logger().info('target lost: searching')
            search_spd = self.search_step()
        self.mode = mode

        self.limit_goals()
        pan_spd, tilt_spd = self.speeds(ff_rate)
        if search_spd is not None:
            pan_spd = search_spd
        self.publish_command(pan_spd, tilt_spd, mode + ('+SWEEP' if self.pan_sweeping else ''))
        self.advance_estimate(pan_spd, tilt_spd, dt)
        self.log_tick(now_s, dt, mode + ('+SWEEP' if self.pan_sweeping else ''), pan_spd, tilt_spd,
                      target_fresh, odom_fresh)

    # ---------------- output ----------------

    def publish_command(self, pan_spd, tilt_spd, tag):
        stamp = self.get_clock().now().to_msg()

        msg = JointState()
        msg.header.stamp = stamp
        msg.name = JOINT_NAMES
        msg.position = [self.pan_goal, self.tilt_goal]
        msg.velocity = [pan_spd, tilt_spd]     # per-axis; firmware takes SX / SY separately
        self.joint_pub.publish(msg)

        est = JointState()
        est.header.stamp = stamp
        est.name = JOINT_NAMES
        est.position = [self.pan_est, self.tilt_est]
        self.estimate_pub.publish(est)

        self.get_logger().info(
            f'[{tag}] goal pan {self.pan_goal:.2f} tilt {self.tilt_goal:.2f} '
            f'spd {pan_spd:.0f}/{tilt_spd:.0f} | est pan {self.pan_est:.2f} tilt {self.tilt_est:.2f}',
            throttle_duration_sec=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = PanTilt()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()