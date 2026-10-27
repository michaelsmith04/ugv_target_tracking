from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ugv_human_detect'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models'), [
            'models/efficientdet_lite0.tflite',
            'models/blaze_face_short_range.tflite',
            'models/face_recognition_sface_2021dec.onnx',
            'models/yolov8n.pt'
        ]),
        (os.path.join('share', package_name, 'audio'), glob('audio/human_detected.wav')),
        (os.path.join('share', package_name, 'reference_faces'), glob('reference_faces/*.jpg')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'human_detector = ugv_human_detect.human_detector:main',
            'pan_tilt_actuation = ugv_human_detect.pan_tilt_actuation:main',
            'lidar_detection = ugv_human_detect.lidar_detection:main',
            'human_transform = ugv_human_detect.humantransform:main',
            'polar_estimator = ugv_human_detect.polarestimator:main',
            'scan_masker = ugv_human_detect.scan_mask:main',
            'scan_resampler = ugv_human_detect.scan_resampler:main',
            'cov_inflator_pt = ugv_human_detect.pose_cov_inflate_pt:main',
            'cov_inflator_oak = ugv_human_detect.pose_cov_inflate_oak:main',
        ],
    },
)
