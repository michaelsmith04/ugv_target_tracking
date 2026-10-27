#!/usr/bin/env python3
"""Turn a tracked target into a standoff goal on /goal_pose.
 
Differences from v1:
  * Subscribes to the *gated measurement* topic as well as the UKF output, so
    it knows when a real detection last arrived (the UKF covariance can't tell
    you that once P has blown up or collapsed).
  * Track confirmation: no goals until `confirm_count` mutually consistent
    detections have been seen (rejects a lone outlier after a dropout).
  * Time-based coast limit: goals stop `coast_time` s after the last detection.
  * Goal-jump rejection: a goal that moves faster than `max_goal_speed` is
    ignored (the allowance grows with time, so it can't lock out forever).
  * Track lost (`lost_timeout` s without detections): forget the last goal,
    publish a "hold here" goal so Nav2 stops, and (optionally) re-seed the UKF
    from the next detection.
 
Run (ROS sourced):
    python3 follow_target.py
    python3 follow_target.py --ros-args -p standoff:=1.5 -p reset_filter:=true
"""
import math
 
import rclpy
import tf2_geometry_msgs  # noqa: F401  (registers PoseStamped support in tf2)
import tf2_ros
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time
 
 
class TrackGate:
    """Pure-python track confirmation logic (no ROS), so it can be unit tested.
 
    A detection continues the current track if it is in the same frame, arrives
    within `lost_timeout` of the previous one, and is not farther from it than
    `max_speed * dt + slack`. Otherwise it starts a new track (count = 1).
    """
 
    def __init__(self, confirm_count, max_speed, slack, lost_timeout):
        self.confirm_count = confirm_count
        self.max_speed = max_speed
        self.slack = slack
        self.lost_timeout = lost_timeout
        self.n = 0
        self.last = None  # (t, frame, x, y)
 
    def update(self, t, frame, x, y):
        """Feed one detection. Returns True if it started a new track."""
        new_track = True
        if self.last is not None:
            lt, lf, lx, ly = self.last
            dt = t - lt
            allowed = self.max_speed * max(dt, 0.05) + self.slack
            if (frame == lf and dt <= self.lost_timeout
                    and math.hypot(x - lx, y - ly) <= allowed):
                new_track = False
        self.n = 1 if new_track else self.n + 1
        self.last = (t, frame, x, y)
        return new_track
 
    def confirmed(self):
        return self.n >= self.confirm_count
 
    def age(self, t):
        return math.inf if self.last is None else t - self.last[0]
 
    def lost(self, t):
        return self.last is not None and self.age(t) > self.lost_timeout
 
    def drop(self):
        self.n = 0
        self.last = None
 
 
class FollowTarget(Node):
    def __init__(self):
        super().__init__('follow_target')
        # --- geometry / rate limiting -------------------------------------
        self.declare_parameter('standoff', 1.0)         # m, stop this far from the person
        self.declare_parameter('hold_band', 0.3)        # m, no goal if already within standoff + band
        self.declare_parameter('min_goal_shift', 0.3)   # m, goal must move this far to be re-sent
        self.declare_parameter('min_period', 1.5)       # s, minimum time between goals
        self.declare_parameter('max_age', 0.0)          # s, ignore older UKF msgs (<= 0 disables)
        self.declare_parameter('max_xy_var', 1.0)       # m^2, secondary gate on x + y variance
        # --- track gating ---------------------------------------------------
        self.declare_parameter('confirm_count', 3)      # consistent detections before following
        self.declare_parameter('max_meas_speed', 3.0)   # m/s, max believable target speed
        self.declare_parameter('meas_slack', 0.3)       # m, added to the speed-based allowance
        self.declare_parameter('coast_time', 2.0)       # s, keep sending goals this long after last detection
        self.declare_parameter('lost_timeout', 3.0)     # s, declare track lost after this
        self.declare_parameter('max_goal_speed', 2.5)   # m/s, goal-jump limit
        self.declare_parameter('goal_jump_margin', 0.5) # m, added to the goal-jump allowance
        self.declare_parameter('stop_on_lost', True)    # publish a hold-here goal when track is lost
        # --- optional UKF re-seed --------------------------------------------
        self.declare_parameter('reset_filter', False)   # re-seed the UKF from the first detection of a new track
        self.declare_parameter('set_pose_topic', '/set_pose')
        self.declare_parameter('reset_variance', 0.5)   # m^2 on x/y when re-seeding
        # --- frames / topics --------------------------------------------------
        self.declare_parameter('goal_frame', 'map')
        self.declare_parameter('robot_frame', 'base_footprint')
        self.declare_parameter('input_topic', '/target/odometry_filtered')
        self.declare_parameter('measurement_topic', '/target/oak/pose_gated')
        self.declare_parameter('output_topic', '/goal_pose')
 
        input_topic = self.p('input_topic')
        meas_topic = self.p('measurement_topic')
        output_topic = self.p('output_topic')
 
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.pub = self.create_publisher(PoseStamped, output_topic, 10)
        self.set_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, self.p('set_pose_topic'), 10)
        self.create_subscription(Odometry, input_topic, self.on_target, 10)
        self.create_subscription(
            PoseWithCovarianceStamped, meas_topic, self.on_measurement, 10)
        self.create_timer(0.2, self.check_lost)
 
        self.gate = TrackGate(self.p('confirm_count'), self.p('max_meas_speed'),
                              self.p('meas_slack'), self.p('lost_timeout'))
        self.needs_reset = True
        self.last_goal = None
        self.last_sent = self.get_clock().now()
        self._last_log = {}
 
        self.get_logger().info(
            f'follow_target: {input_topic} (+{meas_topic}) -> {output_topic}')
 
    # ------------------------------------------------------------------ utils
    def p(self, name):
        return self.get_parameter(name).value
 
    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9
 
    def skip(self, key, why):
        """Log why no goal was sent, at most once per second per reason."""
        t = self.now_s()
        if t - self._last_log.get(key, -1e9) >= 1.0:
            self._last_log[key] = t
            self.get_logger().info(f'no goal [{key}]: {why}')
 
    # ---------------------------------------------------------- measurements
    def on_measurement(self, msg):
        p = msg.pose.pose.position
        new_track = self.gate.update(self.now_s(), msg.header.frame_id, p.x, p.y)
        if new_track:
            self.get_logger().info(
                f'new track at ({p.x:.2f}, {p.y:.2f}) in {msg.header.frame_id}; '
                f'need {self.p("confirm_count")} consistent detections')
            if self.p('reset_filter') and self.needs_reset:
                self.reseed_filter(msg)
                self.needs_reset = False
 
    def reseed_filter(self, meas):
        """Re-initialise the UKF at this detection (UNTESTED against your robot_localization version)."""
        seed = PoseWithCovarianceStamped()
        seed.header = meas.header
        seed.pose.pose = meas.pose.pose
        var = self.p('reset_variance')
        seed.pose.covariance[0] = var
        seed.pose.covariance[7] = var
        seed.pose.covariance[14] = var
        for i in (21, 28, 35):
            seed.pose.covariance[i] = 1e-6
        self.set_pose_pub.publish(seed)
        self.get_logger().info('re-seeded UKF from new track')
 
    def check_lost(self):
        if not self.gate.lost(self.now_s()):
            return
        self.get_logger().warn(
            f'track lost (no detection for > {self.p("lost_timeout"):.1f}s)')
        self.gate.drop()
        self.last_goal = None
        self.needs_reset = True
        if self.p('stop_on_lost'):
            self.hold_here()
 
    def hold_here(self):
        """Publish a goal at the robot's own pose so Nav2 finishes and stops."""
        try:
            r = self.tf_buffer.lookup_transform(
                self.p('goal_frame'), self.p('robot_frame'), Time())
        except tf2_ros.TransformException as e:
            self.get_logger().warn(f'hold goal: TF failed: {e}')
            return
        goal = PoseStamped()
        goal.header.frame_id = self.p('goal_frame')
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = r.transform.translation.x
        goal.pose.position.y = r.transform.translation.y
        goal.pose.orientation = r.transform.rotation
        self.pub.publish(goal)
        self.last_sent = self.get_clock().now()
 
    # ------------------------------------------------------------ UKF output
    def on_target(self, msg):
        now = self.get_clock().now()
        t = now.nanoseconds * 1e-9
 
        # 1. Quality gates.
        max_age = self.p('max_age')
        if max_age > 0 and (now - Time.from_msg(msg.header.stamp)).nanoseconds * 1e-9 > max_age:
            return self.skip('stale', 'UKF message older than max_age')
        if not self.gate.confirmed():
            return self.skip('unconfirmed',
                             f'track not confirmed ({self.gate.n}/{self.p("confirm_count")})')
        age = self.gate.age(t)
        if age > self.p('coast_time'):
            return self.skip('coast', f'{age:.1f}s since last detection > coast_time')
        cov = msg.pose.covariance
        if cov[0] + cov[7] > self.p('max_xy_var'):
            return self.skip('var', f'x+y variance {cov[0] + cov[7]:.2f} > max_xy_var')
        if (now - self.last_sent).nanoseconds * 1e-9 < self.p('min_period'):
            return  # silent: this is the normal rate limit
 
        # 2. Target and robot positions in the goal frame (latest transform).
        frame = self.p('goal_frame')
        ps = PoseStamped()
        ps.header.frame_id = msg.header.frame_id
        ps.pose = msg.pose.pose  # stamp left at 0 -> tf uses the latest transform
        try:
            tgt = self.tf_buffer.transform(ps, frame)
            robot = self.tf_buffer.lookup_transform(frame, self.p('robot_frame'), Time())
        except tf2_ros.TransformException as e:
            return self.skip('tf', f'TF failed: {e}')
 
        rx = robot.transform.translation.x
        ry = robot.transform.translation.y
        tx, ty = tgt.pose.position.x, tgt.pose.position.y
 
        # 3. Dead band: already close enough.
        dist = math.hypot(tx - rx, ty - ry)
        if dist < self.p('standoff') + self.p('hold_band'):
            return
 
        # 4. Standoff goal on the robot -> person line, facing the person.
        yaw = math.atan2(ty - ry, tx - rx)
        gx = tx - self.p('standoff') * math.cos(yaw)
        gy = ty - self.p('standoff') * math.sin(yaw)
 
        # 5. Hysteresis + goal-jump rejection.
        if self.last_goal is not None:
            shift = math.hypot(gx - self.last_goal[0], gy - self.last_goal[1])
            if shift < self.p('min_goal_shift'):
                return
            dt_goal = (now - self.last_sent).nanoseconds * 1e-9
            allowed = self.p('max_goal_speed') * dt_goal + self.p('goal_jump_margin')
            if shift > allowed:
                return self.skip('jump',
                                 f'goal moved {shift:.2f} m in {dt_goal:.1f}s (> {allowed:.2f} m)')
 
        goal = PoseStamped()
        goal.header.frame_id = frame
        goal.header.stamp = now.to_msg()
        goal.pose.position.x = gx
        goal.pose.position.y = gy
        goal.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.orientation.w = math.cos(yaw / 2.0)
        self.pub.publish(goal)
        self.last_goal = (gx, gy)
        self.last_sent = now
 
 
def main():
    rclpy.init()
    node = FollowTarget()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
 
 
if __name__ == '__main__':
    main()
 