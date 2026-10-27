#!/usr/bin/env python3
"""Turn a tracked target into a standoff goal on /goal_pose.

Differences from v1:
  * Subscribes to the *gated measurement* topic as well as the UKF output, so
    it knows when a real detection last arrived (the UKF covariance can't tell
    you that once P has blown up or collapsed).
  * Track confirmation: no goals until `confirm_count` mutually consistent
    detections have been seen (rejects a lone outlier after a dropout).
  * Time-based coast limit: goals stop `coast_time` s after the last detection.
    Nav2 then simply finishes the last goal instead of chasing an extrapolation.
  * Goal-jump rejection: a goal that moves faster than `max_goal_speed` is
    ignored (the allowance grows with time, so it can't lock out forever).
  * Track lost (`lost_timeout` s without detections): forget the last goal,
    optionally cancel the active Nav2 goal (`stop_on_lost`), and (optionally)
    re-seed the UKF from the next detection.

Run (ROS sourced):
    python3 follow_target.py
    python3 follow_target.py --ros-args -p standoff:=1.5 -p reset_filter:=true
"""
import math

import numpy as np
import rclpy
import tf2_geometry_msgs  # noqa: F401  (registers PoseStamped support in tf2)
import tf2_ros

from action_msgs.srv import CancelGoal
from action_msgs.msg import GoalStatusArray

from sensor_msgs.msg import JointState
from vision_msgs.msg import Detection2DArray

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.time import Time


def cell_cost(grid, x, y):
    """Cost at a map point, -1 for unknown, -2 for outside the grid.

    grid = (data[h, w] int array, resolution, origin_x, origin_y)."""
    data, res, ox, oy = grid
    i = int(math.floor((x - ox) / res))
    j = int(math.floor((y - oy) / res))
    if 0 <= i < data.shape[1] and 0 <= j < data.shape[0]:
        return int(data[j, i])
    return -2


def line_max_cost(grid, x0, y0, x1, y1, skip_start=0.0):
    """Max cost along a segment; unknown/outside counts as 100. Cells within
    `skip_start` m of the start are ignored (the rover's own inflated cells)."""
    n = max(2, int(math.hypot(x1 - x0, y1 - y0) / (grid[1] * 0.5)))
    worst = 0
    for k in range(n + 1):
        a = k / n
        x, y = x0 + (x1 - x0) * a, y0 + (y1 - y0) * a
        if math.hypot(x - x0, y - y0) < skip_start:
            continue
        c = cell_cost(grid, x, y)
        worst = max(worst, 100 if c < 0 else c)
    return worst


def select_goal(grid, robot, person, standoff, max_goal_cost=50, max_line_cost=98,
                ring_extra=0.5, step_deg=10.0, max_swing_deg=150.0, skip_start=0.25,
                relax_line=True):
    """Pick a goal near the person that is on a free, known cell.

    The ideal goal is `standoff` m from the person on the person->robot line. If
    that cell is unknown or in/near an obstacle, or the straight line from the
    robot is blocked by lethal/inscribed cells, search a ring around the person
    (radius standoff .. standoff+ring_extra, swinging up to max_swing_deg either
    side) for the feasible cell closest to the ideal one.

    Pass 1 requires a clear straight line from the robot. If nothing qualifies and
    `relax_line` is set, pass 2 only requires a valid goal cell (the planner then
    routes around the obstacle). Returns (x, y, strict) or None.
    """
    rx, ry = robot
    px, py = person
    phi = math.atan2(ry - py, rx - px)        # direction person -> robot
    ideal = (px + standoff * math.cos(phi), py + standoff * math.sin(phi))
    radii = [standoff + ring_extra * k / 2.0 for k in range(3)]
    n = int(max_swing_deg / step_deg)
    for strict in ((True, False) if relax_line else (True,)):
        best, best_score = None, 1e9
        for r in radii:
            for k in range(-n, n + 1):
                th = phi + math.radians(k * step_deg)
                cx, cy = px + r * math.cos(th), py + r * math.sin(th)
                c = cell_cost(grid, cx, cy)
                if c < 0 or c > max_goal_cost:
                    continue
                if strict and line_max_cost(grid, rx, ry, cx, cy, skip_start) > max_line_cost:
                    continue
                score = math.hypot(cx - ideal[0], cy - ideal[1]) + 0.1 * math.hypot(cx - rx, cy - ry)
                if score < best_score:
                    best, best_score = (cx, cy), score
        if best is not None:
            return best[0], best[1], strict
    return None


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


class MultiGate:
    """One TrackGate per measurement source (OAK, pan-tilt, lidar, ...).

    Each source is checked for self-consistency in its own frame, so sensors
    that publish in different frames can't reset each other's count. The
    track counts as confirmed/fresh if ANY source is.
    """

    def __init__(self, sources, confirm_count, max_speed, slack, lost_timeout):
        self.gates = {s: TrackGate(confirm_count, max_speed, slack, lost_timeout)
                      for s in sources}

    def any_active(self, t):
        return any(g.last is not None and not g.lost(t) for g in self.gates.values())

    def update(self, source, t, frame, x, y):
        """Returns (new_track_for_source, whole_track_was_inactive)."""
        was_inactive = not self.any_active(t)
        return self.gates[source].update(t, frame, x, y), was_inactive

    def freshest(self, t, max_age):
        """(source, age) of the best confirmed source seen within max_age, else None."""
        best = None
        for name, g in self.gates.items():
            if g.confirmed() and g.age(t) <= max_age and (best is None or g.age(t) < best[1]):
                best = (name, g.age(t))
        return best

    def status(self, t):
        return ', '.join(f'{n}: {g.n} det, {g.age(t):.1f}s' for n, g in self.gates.items())

    def all_lost(self, t):
        seen = [g for g in self.gates.values() if g.last is not None]
        return bool(seen) and all(g.lost(t) for g in seen)

    def drop_all(self):
        for g in self.gates.values():
            g.drop()


class FollowTarget(Node):
    def __init__(self):
        super().__init__('follow_target')
        # --- geometry / rate limiting -------------------------------------
        self.declare_parameter('standoff', 0.5)         # m, stop this far from the person
        self.declare_parameter('hold_band', 0.3)        # m, no goal if already within standoff + band
        self.declare_parameter('min_goal_shift', 0.3)   # m, goal must move this far to be re-sent
        self.declare_parameter('min_period', 1.5)       # s, minimum time between goals
        self.declare_parameter('max_age', 0.0)          # s, ignore older UKF msgs (<= 0 disables)
        self.declare_parameter('max_xy_var', 1.0)       # m^2, secondary gate on x + y variance
        # --- track gating ---------------------------------------------------
        self.declare_parameter('confirm_count', 3)      # consistent detections before following
        self.declare_parameter('max_meas_speed', 3.0)   # m/s, max believable target speed
        self.declare_parameter('meas_slack', 0.3)       # m, added to the speed-based allowance
        self.declare_parameter('coast_time', 0.5)       # s, keep sending goals this long after last detection (UKF velocity is too noisy to steer on for longer)
        self.declare_parameter('lost_timeout', 3.0)     # s, declare track lost after this
        self.declare_parameter('max_yaw_rate_for_goal', 2.0)  # rad/s, no new goals while the rover turns faster (<=0 disables); keep ABOVE the rotation speed cap
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('use_costmap', True)      # choose goals on free, known, reachable cells
        self.declare_parameter('costmap_topic', '/global_costmap/costmap')
        #self.declare_parameter('max_goal_cost', 50)      # occupancy value (0-100) allowed at the goal cell
        self.declare_parameter('max_goal_cost', 98)      # occupancy value (0-100) allowed at the goal cell
        self.declare_parameter('max_line_cost', 98)      # >=99 = inscribed/lethal on the straight line
        self.declare_parameter('ring_extra', 0.5)        # m, search radii standoff .. standoff + ring_extra
        self.declare_parameter('max_goal_speed', 2.5)   # m/s, goal-jump limit
        self.declare_parameter('goal_jump_margin', 0.5) # m, added to the goal-jump allowance
        self.declare_parameter('stop_on_lost', False)   # cancel the active Nav2 goal when track is lost (default: let Nav2 finish the last goal)
        # --- optional UKF re-seed --------------------------------------------
        self.declare_parameter('reset_filter', False)   # re-seed the UKF from the first detection of a new track
        self.declare_parameter('freeze_on_lost', False) # when the track is lost, re-seed the UKF at the last good position (zero velocity) so it stops running away
        self.declare_parameter('set_pose_topic', '/set_pose')
        self.declare_parameter('reset_variance', 0.5)   # m^2 on x/y when re-seeding
        # --- approach parameters ----------------------------------------------
        self.declare_parameter('approach_overshoot', 0.4)  # m, aim Nav2 goal this far inside standoff
        self.declare_parameter('stop_margin', 0.1)         # m, begin braking at standoff + margin
        self.declare_parameter('face_tolerance', 0.15)     # rad, stop rotating inside this
        self.declare_parameter('face_gain', 1.5)
        self.declare_parameter('face_max_w', 0.6)          # rad/s
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        # --- face parameters --------------------------------------------------
        self.declare_parameter('face_trigger', 0.35)    # rad (~20 deg): start turning when off-axis by this much
        self.declare_parameter('nav_status_topic', '/navigate_to_pose/_action/status')

        self.declare_parameter('bearing_face', True)
        self.declare_parameter('joint_est_topic', '/pantilt/joint_estimate')
        self.declare_parameter('pt_det_topic', '/pantilt/detections')
        self.declare_parameter('bearing_det_age', 0.4)    # s, detection must be this fresh
        self.declare_parameter('bearing_trigger', 0.35)   # rad, start turning beyond this
        self.declare_parameter('bearing_tol', 0.12)       # rad, stop inside this
        # --- frames / topics --------------------------------------------------
        self.declare_parameter('goal_frame', 'map')
        self.declare_parameter('robot_frame', 'base_footprint')
        self.declare_parameter('input_topic', '/target/odometry_filtered')
        self.declare_parameter('measurement_topics',
                              # ['/target/oak/pose_gated', '/pantilt/pose_estimates'])
                                ['/target/oak/pose_gated', '/pantilt/pose_inflated'])
        self.declare_parameter('output_topic', '/goal_pose')

        input_topic = self.p('input_topic')
        meas_topics = list(self.p('measurement_topics'))
        output_topic = self.p('output_topic')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.pub = self.create_publisher(PoseStamped, output_topic, 10)
        self.set_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, self.p('set_pose_topic'), 10)
        self.create_subscription(Odometry, input_topic, self.on_target, 10)
        for topic in meas_topics:
            self.create_subscription(
                PoseWithCovarianceStamped, topic,
                lambda msg, topic=topic: self.on_measurement(topic, msg), 10)
        self.yaw_rate = 0.0
        self.grid = None
        self.create_subscription(OccupancyGrid, self.p('costmap_topic'), self.on_costmap, 1)
        self.create_subscription(Odometry, self.p('odom_topic'), self.on_odom, 20)
        self.cancel_cli = self.create_client(CancelGoal, '/navigate_to_pose/_action/cancel_goal')
        self.create_timer(0.2, self.check_lost)

        self.pan_est = 0.0
        self.det_t = -1e9
        self.ukf_ok_t = -1e9
        self.bearing_active = False
        self.create_subscription(JointState, self.p('joint_est_topic'), self.on_joint_est, 10)
        self.create_subscription(Detection2DArray, self.p('pt_det_topic'), self.on_pt_det, 10)
        self.create_timer(0.05, self.bearing_tick)

        self.cmd_pub = self.create_publisher(Twist, self.p('cmd_vel_topic'), 10)
        self.holding = False

        self.gate = MultiGate(meas_topics, self.p('confirm_count'), self.p('max_meas_speed'),
                              self.p('meas_slack'), self.p('lost_timeout'))
        self.needs_reset = True
        self.last_good = None   # (frame_id, x, y) of the UKF estimate while a confirmed source was fresh
        self.last_goal = None
        self.last_sent = self.get_clock().now()
        self._last_log = {}

        self.nav_busy = False
        self.facing = False
        self.create_subscription(GoalStatusArray, self.p('nav_status_topic'), self.on_nav_status, 10)

        self.get_logger().info(
            f'follow_target with static turning: {input_topic} (+{meas_topics}) -> {output_topic}')

    # ------------------------------------------------------------------ utils
    def on_nav_status(self, msg):
        # 1 = accepted, 2 = executing, 3 = canceling
        self.nav_busy = any(s.status in (1, 2, 3) for s in msg.status_list)

    def nav_active(self, now):
        # also treat as busy for 0.5 s after we send a goal, until the status arrives
        return self.nav_busy or (now - self.last_sent).nanoseconds * 1e-9 < 0.5

    def face_target(self, rx, ry, ryaw, tx, ty, trigger):
        err = math.atan2(ty - ry, tx - rx) - ryaw
        err = math.atan2(math.sin(err), math.cos(err))
        if self.facing and abs(err) < self.p('face_tolerance'):
            self.facing = False
        elif not self.facing and abs(err) > trigger:
            self.facing = True
        cmd = Twist()
        if self.facing:
            wmax = self.p('face_max_w')
            cmd.angular.z = max(-wmax, min(wmax, self.p('face_gain') * err))
        return cmd

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

    def on_costmap(self, msg):
        data = np.array(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
        self.grid = (data, msg.info.resolution, msg.info.origin.position.x,
                     msg.info.origin.position.y)

    def on_odom(self, msg):
        self.yaw_rate = msg.twist.twist.angular.z

    def on_joint_est(self, msg):
        self.pan_est = msg.position[0]

    def on_pt_det(self, msg):
        if msg.detections:
            self.det_t = self.now_s()

    def bearing_tick(self):
        if not self.p('bearing_face'):
            return
            
        now = self.now_s()
        # Only when the UKF-based logic isn't driving things
        ukf_ok = now - self.ukf_ok_t < 0.5
        seen = now - self.det_t < self.p('bearing_det_age')
        if ukf_ok or not seen or self.nav_active(self.get_clock().now()):
            if self.bearing_active:                  # stop once, never leave a stale command
                self.cmd_pub.publish(Twist())
                self.bearing_active = False
            return
        err = -self.pan_est                          # +pan = clockwise, +angular.z = CCW
        if self.bearing_active and abs(err) < self.p('bearing_tol'):
            self.bearing_active = False
            self.cmd_pub.publish(Twist())
            return
        if not self.bearing_active and abs(err) > self.p('bearing_trigger'):
            self.bearing_active = True
        if self.bearing_active:
            cmd = Twist()
            wmax = self.p('face_max_w')
            cmd.angular.z = max(-wmax, min(wmax, self.p('face_gain') * err))
            self.cmd_pub.publish(cmd)

    # ---------------------------------------------------------- measurements
    def on_measurement(self, source, msg):
        p = msg.pose.pose.position
        new_track, was_inactive = self.gate.update(
            source, self.now_s(), msg.header.frame_id, p.x, p.y)
        if new_track:
            self.get_logger().info(
                f'[{source}] new track at ({p.x:.2f}, {p.y:.2f}) in {msg.header.frame_id}; '
                f'need {self.p("confirm_count")} consistent detections')
            if was_inactive and self.p('reset_filter') and self.needs_reset:
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
        if not self.gate.all_lost(self.now_s()):
            return
        self.get_logger().warn(
            f'track lost (no source for > {self.p("lost_timeout"):.1f}s)')
        self.gate.drop_all()
        self.last_goal = None
        self.needs_reset = True
        if self.p('freeze_on_lost') and self.last_good is not None:
            frame, x, y = self.last_good
            seed = PoseWithCovarianceStamped()
            seed.header.frame_id = frame
            seed.header.stamp = self.get_clock().now().to_msg()
            seed.pose.pose.position.x = x
            seed.pose.pose.position.y = y
            seed.pose.pose.orientation.w = 1.0
            var = self.p('reset_variance')
            seed.pose.covariance[0] = seed.pose.covariance[7] = seed.pose.covariance[14] = var
            for i in (21, 28, 35):
                seed.pose.covariance[i] = 1e-6
            self.set_pose_pub.publish(seed)
            self.get_logger().info(f'froze UKF at last good position ({x:.2f}, {y:.2f})')
        if self.p('stop_on_lost'):
            self.cancel_nav()

    def cancel_nav(self):
        """Cancel all active NavigateToPose goals (empty goal_info = cancel all).

        Deliberately NOT a 'goal at my own pose': in the second recording that
        made NavFn return a degenerate plan with an arbitrary first heading and
        the rotation shim spun the rover ~180 deg to face it.
        """
        if not self.cancel_cli.service_is_ready():
            self.get_logger().warn('cancel_goal service not available')
            return
        self.cancel_cli.call_async(CancelGoal.Request())

    # ------------------------------------------------------------ UKF output
    def on_target(self, msg):
        now = self.get_clock().now()
        t = now.nanoseconds * 1e-9

        # 1. Quality gates.
        max_age = self.p('max_age')
        if max_age > 0 and (now - Time.from_msg(msg.header.stamp)).nanoseconds * 1e-9 > max_age:
            return self.skip('stale', 'UKF message older than max_age')
        if self.gate.freshest(t, self.p('coast_time')) is None:
            return self.skip(
                'track', f'no confirmed source within coast_time '
                         f'({self.gate.status(t)}; need {self.p("confirm_count")} det)')
        if msg.pose.covariance[0] + msg.pose.covariance[7] <= self.p('max_xy_var'):
            self.last_good = (msg.header.frame_id, msg.pose.pose.position.x, msg.pose.pose.position.y)
        lim = self.p('max_yaw_rate_for_goal')
        if lim > 0 and abs(self.yaw_rate) > lim:
            return self.skip('turning', f'rover yaw rate {self.yaw_rate:.2f} rad/s > {lim:.1f}; '
                                        'target estimates are least reliable while turning')
        cov = msg.pose.covariance
        if cov[0] + cov[7] > self.p('max_xy_var'):
            return self.skip('var', f'x+y variance {cov[0] + cov[7]:.2f} > max_xy_var')
        if (now - self.last_sent).nanoseconds * 1e-9 < self.p('min_period'):
            return  # silent: this is the normal rate limit

        self.ukf_ok_t = t

         # 2. Positions in the goal frame
        frame = self.p('goal_frame')
        ps = PoseStamped()
        ps.header.frame_id = msg.header.frame_id
        ps.pose = msg.pose.pose
        try:
            tgt = self.tf_buffer.transform(ps, frame)
            robot = self.tf_buffer.lookup_transform(frame, self.p('robot_frame'), Time())
        except tf2_ros.TransformException as e:
            return self.skip('tf', f'TF failed: {e}')
        rx, ry = robot.transform.translation.x, robot.transform.translation.y
        q = robot.transform.rotation
        ryaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        tx, ty = tgt.pose.position.x, tgt.pose.position.y
        dist = math.hypot(tx - rx, ty - ry)

        # 3. Hold state with hysteresis: stop + face. Runs BEFORE rate/turn gates.
        enter = self.p('standoff') + self.p('stop_margin')
        leave = self.p('standoff') + self.p('hold_band')
        if self.holding and dist > leave:
            self.holding = False
        elif not self.holding and dist <= enter:
            self.holding = True
            self.last_goal = None
            self.cancel_nav()
            self.get_logger().info(f'within {dist:.2f} m: cancelled Nav2, facing target')
        if self.holding:
            self.cmd_pub.publish(self.face_target(rx, ry, ryaw, tx, ty, self.p('face_tolerance')))
            return

        busy = self.nav_active(now)
        if busy:
            self.facing = False
        else:
            # Nav2 is idle: turn toward the target if we're off-axis, even when no new goal is due
            cmd = self.face_target(rx, ry, ryaw, tx, ty, self.p('face_trigger'))
            if self.facing:
                self.cmd_pub.publish(cmd)    

        # 4. Rate limits (only apply while approaching)
        lim = self.p('max_yaw_rate_for_goal')
        if lim > 0 and abs(self.yaw_rate) > lim:
            return self.skip('turning', f'yaw rate {self.yaw_rate:.2f} > {lim:.1f}')
        if (now - self.last_sent).nanoseconds * 1e-9 < self.p('min_period'):
            return

        

        # 5. Goal inside the standoff so Nav2 is still moving when we cancel it
        aim = max(self.p('standoff') - self.p('approach_overshoot'), 0.6)
        yaw = math.atan2(ty - ry, tx - rx)
        gx, gy = tx - aim * math.cos(yaw), ty - aim * math.sin(yaw)
        if self.p('use_costmap'):
            if self.grid is None:
                return self.skip('costmap', 'no costmap received yet')
            pick = select_goal(self.grid, (rx, ry), (tx, ty), aim,
                            self.p('max_goal_cost'), self.p('max_line_cost'),
                            self.p('ring_extra'))
            if pick is None:
                return self.skip('nogoal', 'no free, reachable cell near the target')
            gx, gy, strict = pick
            yaw = math.atan2(ty - gy, tx - gx)

        # 6. Hysteresis + goal-jump rejection.
        if self.last_goal is not None and busy:
            shift = math.hypot(gx - self.last_goal[0], gy - self.last_goal[1])
            if shift < self.p('min_goal_shift'):
                return
            dt_goal = (now - self.last_sent).nanoseconds * 1e-9
            allowed = self.p('max_goal_speed') * dt_goal + self.p('goal_jump_margin')
            if shift > allowed:
                return self.skip('jump', f'goal moved {shift:.2f} m in {dt_goal:.1f}s (> {allowed:.2f} m)')

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