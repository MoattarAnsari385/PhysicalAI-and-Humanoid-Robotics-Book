from setuptools import setup
import os
from glob import glob

package_name = 'minimal_robot_package'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*launch.[pxy][yma]*')),
        # Include all config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        # Include all URDF files
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Textbook Maintainer',
    maintainer_email='textbook@example.com',
    description='A minimal robot package example for the Physical AI & Humanoid Robotics textbook',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'minimal_publisher = minimal_robot_package.minimal_publisher:main',
            'minimal_subscriber = minimal_robot_package.minimal_subscriber:main',
            'robot_controller = minimal_robot_package.robot_controller:main',
        ],
    },
)