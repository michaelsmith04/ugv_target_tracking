#!/usr/bin/env python3
"""
Converts DR-SPAAM's geometry_msgs/PoseArray detections into a single
geometry_msgs/PoseWithCovarianceStamped for consumption by robot_localization's
ukf_node.

PoseArray carries no confidence score and no per-pose covariance, so:
  - Detection selection defaults to the first pose in the array (single-person
    assumption for now -- see select_detection() to extend this later).
  - Covariance is supplied entirely via ROS parameters, since DR-SPAAM gives
    you nothing to derive it from. These are PLACEHOLDER values until you run
    the empirical calibration (known-distance test, sample variance).

z is intentionally left at 0.0 and its covariance set high, since a 2D LiDAR
gives no elevation information. This is fine because pose0_config in the UKF
yaml masks z out for this sensor anyway -- the value is never read.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, PoseWithCovarianceStamped


class LidarPoseConverter(Node):
    def __init__(self):
        super().__init__('lidar_pose_converter')

        # --- Placeholder covariance parameters (tune via calibration) ---
        self.declare_parameter('var_x', 0.05)     # m^2
        self.declare_parameter('var_y', 0.05)     # m^2
        self.declare_parameter('var_z', 1.0e3)    # unused (masked), keep large
        self.declare_parameter('var_orientation', 1.0e3)  # unused, masked

        self.declare_parameter('input_topic', '/dr_spaam_detections')
        self.declare_parameter('output_topic', '/target/lidar/pose_gated')

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value

        self.sub = self.create_subscription(
            PoseArray, input_topic, self.callback, 10)
        self.pub = self.create_publisher(
            PoseWithCovarianceStamped, output_topic, 10)

        self.get_logger().info(
            f'lidar_pose_converter: {input_topic} -> {output_topic}')

    def select_detection(self, msg: PoseArray):
        """Pick which detection to track. Single-person placeholder: first
        pose in the array. Extend this to nearest-to-prediction or highest
        confidence once you support multi-person scenes or add a score
        source."""
        if len(msg.poses) == 0:
            return None
        return msg.poses[0]

    def callback(self, msg: PoseArray):
        pose = self.select_detection(msg)
        if pose is None:
            return

        out = PoseWithCovarianceStamped()
        out.header.stamp = msg.header.stamp  # preserve original detection time
        out.header.frame_id = msg.header.frame_id

        out.pose.pose.position.x = pose.position.x
        out.pose.pose.position.y = pose.position.y
        out.pose.pose.position.z = 0.0  # no elevation from 2D LiDAR

        out.pose.pose.orientation.w = 1.0  # unused, masked in UKF config

        var_x = self.get_parameter('var_x').value
        var_y = self.get_parameter('var_y').value
        var_z = self.get_parameter('var_z').value
        var_o = self.get_parameter('var_orientation').value

        # Row-major 6x6: x, y, z, roll, pitch, yaw
        cov = [0.0] * 36
        cov[0] = var_x
        cov[7] = var_y
        cov[14] = var_z
        cov[21] = var_o
        cov[28] = var_o
        cov[35] = var_o
        out.pose.covariance = cov

        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = LidarPoseConverter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
