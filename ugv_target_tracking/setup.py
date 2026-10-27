import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'ugv_target_tracking'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='you@example.com',
    description='Sensor conversion nodes and UKF config for single-target tracking',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lidar_pose_converter = ugv_target_tracking.lidar_pose_converter:main',
            'oak_pose_converter = ugv_target_tracking.oak_pose_converter:main',
            'follow_target = ugv_target_tracking.follow_target:main',
        ],
    },
)
