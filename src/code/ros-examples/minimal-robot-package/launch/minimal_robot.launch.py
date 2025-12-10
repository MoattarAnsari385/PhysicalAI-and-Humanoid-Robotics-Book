from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_dir = get_package_share_directory('minimal_robot_package')

    return LaunchDescription([
        # Robot State Publisher node to publish URDF
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{
                'robot_description': open(
                    os.path.join(pkg_dir, 'urdf', 'humanoid_skeleton.urdf')
                ).read()
            }]
        ),

        # Minimal publisher node
        Node(
            package='minimal_robot_package',
            executable='minimal_publisher',
            name='minimal_publisher',
            output='screen'
        ),

        # Minimal subscriber node
        Node(
            package='minimal_robot_package',
            executable='minimal_subscriber',
            name='minimal_subscriber',
            output='screen'
        ),

        # Robot controller node with parameters
        Node(
            package='minimal_robot_package',
            executable='robot_controller',
            name='robot_controller',
            parameters=[os.path.join(pkg_dir, 'config', 'robot_params.yaml')],
            output='screen'
        )
    ])