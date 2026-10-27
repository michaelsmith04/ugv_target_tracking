from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('ugv_target_tracking')
    ukf_config = os.path.join(pkg_share, 'config', 'ukf_target_tracking.yaml')

    return LaunchDescription([
        #Node(
        #    package='ugv_target_tracking',
        #    executable='lidar_pose_converter',
        #    name='lidar_pose_converter',
        #    output='screen',
        #),
        Node(
            package='ugv_target_tracking',
            executable='oak_pose_converter',
            name='oak_pose_converter',
            output='screen',
        ),
        Node(
            package='ugv_target_tracking',
            executable='follow_target',
            name='follow_target',
            output='screen',
        ),
        Node(
            package='robot_localization',
            executable='ukf_node',
            name='ukf_filter_node',
            output='screen',
            parameters=[ukf_config],
            remappings=[
                ('/odometry/filtered', '/target/odometry_filtered'),
            ],
        ),
    ])
