#!/usr/bin/env python3
import math
 
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, CameraInfo
from nav_msgs.msg import Odometry
from vision_msgs.msg import Detection2DArray
 
JOINT_NAMES = ['pt_base_link_to_pt_link1', 'pt_link1_to_pt_link2']
 
 
def clamp(x, lo, hi):
    return max(lo, min(hi, x))
 
 
class PanTilt(Node):
    def __init__(self):
        super().__init__('pan_tilt')
 
        self.odom_sub = self.create_subscription(
            Odometry, 'target/odometry_filtered', self.odometry_callback, 10
        )
        self.detection_sub = self.create_subscription(
            Detection2DArray, 'pantilt/detections', self.pt_detection_callback, 10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, 'pantilt/camera_info', self.camera_info_callback, 10
        )
        # NOTE: no subscription to /ugv/joint_states any more. That topic is just an
        # echo of ugv/pt_cmd (via joint_state_publisher), so it carries no real feedback.
 
        self.joint_pub = self.create_publisher(JointState, 'ugv/pt_cmd', 10)
        # Our own best estimate of where the camera is (for the world-frame projection node)
        self.estimate_pub = self.create_publisher(JointState, 'pantilt/joint_estimate', 10)
 
        # --- UKF fallback inputs ---
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_z = 0.0
        self.ENABLE_UKF_FALLBACK = True   # False = hold position when there is no detection
 
        # --- pseudo-position: what we believe the camera angle is (rad) ---
        # Always equal to the last goal we published, and that goal never moves further
        # per tick than the servo can travel, so it stays close to reality.
        self.pan_est = 0.0
        self.tilt_est = 0.0
 
        self.PAN_LIMIT = math.pi        # placeholder
        self.TILT_LIMIT = math.pi / 2   # placeholder
 
        # Max travel speed of each axis (rad/s). MEASURE ON HARDWARE, and err on the slow side.
        self.MAX_PAN_RATE = 2.5         # placeholder
        self.MAX_TILT_RATE = 2.5        # placeholder
 
        # Command (0, 0) for this long at startup so the estimate and the camera agree
        self.HOMING_SEC = 3.0
 
        # --- detection / IBVS state ---
        self.image_width = 640    # overwritten once camera_info arrives
        self.image_height = 480
        self.bbox_center_x = None
        self.bbox_center_y = None
        self.has_target = False
        self.tracking = False     # True while we are continuously in IBVS mode
        self.prev_pixel_error_x = 0.0
        self.last_detection_time = self.get_clock().now()
        self.DETECTION_TIMEOUT_SEC = 0.5
 
        # IBVS gains: radians of goal change per tick, per pixel of error (tune on hardware)
        #self.KP_PAN = 0.000125
        #self.KP_TILT = 0.000125
        self.KP_PAN = 0.00035
        self.KP_TILT = 0.0003
        self.KD_PAN = 0.000025

        self.PIXEL_DEADBAND = 8
        self.MAX_LEAD = 1   # rad the goal may run ahead of the estimate (tune)
        self.SPD_MIN = 1200.0
        self.SPD_GAIN = 300.0
 
        self.CONTROL_PERIOD_SEC = 0.05  # 20 Hz
        self.start_time = self.get_clock().now()
        self.last_loop_time = self.start_time
        self.control_timer = self.create_timer(self.CONTROL_PERIOD_SEC, self.control_loop)
 
    # ---------------- subscriptions ----------------
 
    def camera_info_callback(self, msg):
        self.image_width = msg.width
        self.image_height = msg.height
 
    def odometry_callback(self, msg):
        self.odom_x = msg.pose.pose.position.x  # horizontal, right positive
        self.odom_y = msg.pose.pose.position.y  # forward positive
        self.odom_z = msg.pose.pose.position.z  # upward positive
 
    def pt_detection_callback(self, msg):
        self.last_detection_time = self.get_clock().now()
 
        if not msg.detections:
            self.has_target = False
            return
 
        best = max(
            msg.detections,
            key=lambda d: d.results[0].hypothesis.score if d.results else 0.0,
        )
        self.bbox_center_x = best.bbox.center.position.x
        self.bbox_center_y = best.bbox.center.position.y
        self.has_target = True
 
    # ---------------- control loop ----------------
 
    def control_loop(self):
        now = self.get_clock().now()
        dt = (now - self.last_loop_time).nanoseconds / 1e9
        self.last_loop_time = now
        dt = clamp(dt, 0.0, 0.2)   # guard against scheduling hiccups
 
        # Homing: hold (0, 0) until the camera has certainly arrived there
        if (now - self.start_time).nanoseconds / 1e9 < self.HOMING_SEC:
            self.publish_command(0.0, 0.0, 0.0)
            return
 
        seconds_since_detection = (now - self.last_detection_time).nanoseconds / 1e9
        detection_is_fresh = seconds_since_detection <= self.DETECTION_TIMEOUT_SEC
 
        if self.has_target and detection_is_fresh:
            self.step_ibvs(dt)
        else:
            self.tracking = False   # so the D term doesn't kick when we reacquire
            if self.ENABLE_UKF_FALLBACK:
                self.step_ukf(dt)
 
    def step_ibvs(self, dt):
        pixel_error_x = self.bbox_center_x - self.image_width / 2.0
        pixel_error_y = self.bbox_center_y - self.image_height / 2.0
 
        if abs(pixel_error_x) < self.PIXEL_DEADBAND:
            pixel_error_x = 0.0
        if abs(pixel_error_y) < self.PIXEL_DEADBAND:
            pixel_error_y = 0.0
 
        # First tick after (re)acquiring: no meaningful previous error
        if not self.tracking:
            self.prev_pixel_error_x = pixel_error_x
            self.tracking = True
 
        pan_error_delta = pixel_error_x - self.prev_pixel_error_x
        self.prev_pixel_error_x = pixel_error_x
 
        # Increment from our own estimate, not from feedback
        pan_target = self.pan_est + self.KP_PAN * pixel_error_x - self.KD_PAN * pan_error_delta
        # image y grows downward; confirm sign against your tilt convention on hardware
        tilt_target = self.tilt_est - self.KP_TILT * pixel_error_y

        pan_tilt_velocity = max(self.SPD_MIN, self.SPD_GAIN*math.hypot(pixel_error_x, pixel_error_y))
 
        self.move_toward(pan_target, tilt_target, pan_tilt_velocity, dt, 'IBVS')
 
    def step_ukf(self, dt):
        pan_target = -math.atan2(self.odom_y, self.odom_x)
        tilt_target = math.atan2(self.odom_z, math.hypot(self.odom_x, self.odom_y))
        #self.move_toward(pan_target, tilt_target, dt, 'UKF')
 
    # ---------------- helpers ----------------
 
    def slew(self, target, est, max_rate, dt):
        step = max_rate * dt
        return est + clamp(target - est, -step, step)
 
    def move_toward(self, pan_target, tilt_target, pan_tilt_velocity, dt, tag):
        pan_target = clamp(pan_target, -self.PAN_LIMIT, self.PAN_LIMIT)
        tilt_target = clamp(tilt_target, -self.TILT_LIMIT, self.TILT_LIMIT)

        # goal: may lead the estimate, so the servo can reach cruise speed
        pan_goal = clamp(pan_target, self.pan_est - self.MAX_LEAD, self.pan_est + self.MAX_LEAD)
        tilt_goal = clamp(tilt_target, self.tilt_est - self.MAX_LEAD, self.tilt_est + self.MAX_LEAD)
        self.publish_command(pan_goal, tilt_goal, pan_tilt_velocity)

        # estimate: chases the goal at the servo's measured speed
        self.pan_est = self.slew(pan_goal, self.pan_est, self.MAX_PAN_RATE, dt)
        self.tilt_est = self.slew(tilt_goal, self.tilt_est, self.MAX_TILT_RATE, dt)
 
    def publish_command(self, pan, tilt, velocity):
        stamp = self.get_clock().now().to_msg()

        msg = JointState()
        msg.header.stamp = stamp
        msg.name = JOINT_NAMES
        msg.position = [pan, tilt]                       # goal sent to the servo
        #msg.position = [1.57, 0.0]
        msg.velocity = [velocity, velocity]
        self.joint_pub.publish(msg)

        est = JointState()
        est.header.stamp = stamp
        est.name = JOINT_NAMES
        est.position = [self.pan_est, self.tilt_est]     # our estimate of the actual angle
        self.estimate_pub.publish(est)

        self.get_logger().info(
            f'goal pan {pan:.2f} tilt {tilt:.2f} spd {velocity:.0f} | '
            f'est pan {self.pan_est:.2f} tilt {self.tilt_est:.2f}',
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