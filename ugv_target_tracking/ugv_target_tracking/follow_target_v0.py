#!/usr/bin/env python3
"""Turn /target/odometry_filtered into a standoff goal on /goal_pose.
 
Run (ROS sourced):
    python3 follow_target.py
    python3 follow_target.py --ros-args -p standoff:=1.5 -p max_age:=0.0
"""
import math
 
import rclpy
import tf2_geometry_msgs  # noqa: F401  (registers PoseStamped support in tf2)
import tf2_ros
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time
 
 
class FollowTarget(Node):
    def __init__(self):
        super().__init__('follow_target')
        self.declare_parameter('standoff', 1.0)         # m, stop this far from the person
        self.declare_parameter('hold_band', 0.3)        # m, no goal if already within standoff + band
        self.declare_parameter('min_goal_shift', 0.1)   # m, goal must move this far to be re-sent
        self.declare_parameter('min_period', 1.0)       # s, minimum time between goals
        self.declare_parameter('max_age', 0.0)          # s, ignore older messages (<= 0 disables)
        self.declare_parameter('max_xy_var', 3)       # m^2, ignore if x + y variance exceeds this
        self.declare_parameter('goal_frame', 'map')
        self.declare_parameter('robot_frame', 'base_footprint')
 
        self.declare_parameter('input_topic', '/target/odometry_filtered')
        self.declare_parameter('output_topic', '/goal_pose')

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.pub = self.create_publisher(PoseStamped, output_topic, 10)
        self.create_subscription(Odometry, input_topic, self.on_target, 10)
 
        self.last_goal = None
        self.last_sent = self.get_clock().now()

        self.get_logger().info(
            f'target_follower: {input_topic} -> {output_topic}')
 
    def p(self, name):
        return self.get_parameter(name).value
 
    def on_target(self, msg):
        now = self.get_clock().now()
 
        # 1. Quality gates: stale data, high uncertainty, rate limit.
        max_age = self.p('max_age')
        if max_age > 0 and (now - Time.from_msg(msg.header.stamp)).nanoseconds * 1e-9 > max_age:
            return
        cov = msg.pose.covariance
        if cov[0] + cov[7] > self.p('max_xy_var'):
            return
        if (now - self.last_sent).nanoseconds * 1e-9 < self.p('min_period'):
            return
 
        # 2. Target and robot positions in the goal frame (latest available transform).
        frame = self.p('goal_frame')
        ps = PoseStamped()
        ps.header.frame_id = msg.header.frame_id
        ps.pose = msg.pose.pose  # stamp left at 0 -> tf uses the latest transform
        try:
            tgt = self.tf_buffer.transform(ps, frame)
            robot = self.tf_buffer.lookup_transform(frame, self.p('robot_frame'), Time())
        except tf2_ros.TransformException as e:
            self.get_logger().warn(f'TF failed: {e}', throttle_duration_sec=2.0)
            return
 
        rx = robot.transform.translation.x
        ry = robot.transform.translation.y
        tx, ty = tgt.pose.position.x, tgt.pose.position.y
 
        # 3. Dead band: already close enough, send nothing.
        dist = math.hypot(tx - rx, ty - ry)
        if dist < self.p('standoff') + self.p('hold_band'):
            return
 
        # 4. Standoff goal on the robot -> person line, facing the person.
        yaw = math.atan2(ty - ry, tx - rx)
        gx = tx - self.p('standoff') * math.cos(yaw)
        gy = ty - self.p('standoff') * math.sin(yaw)
 
        # 5. Hysteresis: only re-send if the goal moved enough.
        if self.last_goal is not None:
            shift = math.hypot(gx - self.last_goal[0], gy - self.last_goal[1])
            if shift < self.p('min_goal_shift'):
                return
 
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
 