---
title: "Building a Minimal Robot Package"
sidebar_position: 5
description: "Step-by-step guide to creating a complete ROS 2 robot package with URDF, nodes, and launch files"
---

# Building a Minimal Robot Package

## Introduction

This section provides a comprehensive guide to creating a complete ROS 2 robot package. We'll build a humanoid robot package that demonstrates all the concepts learned in this module: nodes, topics, services, actions, URDF, and proper package structure.

## Package Structure Overview

A proper ROS 2 package follows this structure:

```
minimal_robot_package/
├── CMakeLists.txt          # Build configuration (for C++)
├── package.xml            # Package metadata
├── setup.py              # Python build configuration
├── setup.cfg             # Installation configuration
├── resource/             # Resource files
│   └── minimal_robot_package
├── minimal_robot_package/ # Python module
│   ├── __init__.py
│   ├── minimal_publisher.py
│   ├── minimal_subscriber.py
│   └── robot_controller.py
├── launch/               # Launch files
│   └── minimal_robot.launch.py
├── config/               # Configuration files
├── urdf/                 # Robot description files
│   └── humanoid_skeleton.urdf
├── meshes/               # 3D mesh files
├── worlds/               # Simulation worlds
└── test/                 # Test files
```

## Creating the Package Files

### 1. Package Metadata (package.xml)

The `package.xml` file provides metadata about the package:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>minimal_robot_package</name>
  <version>1.0.0</version>
  <description>A minimal robot package example for the Physical AI & Humanoid Robotics textbook</description>
  <maintainer email="textbook@example.com">Textbook Maintainer</maintainer>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>nav_msgs</depend>

  <exec_depend>ros2launch</exec_depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

### 2. Python Setup (setup.py)

The `setup.py` file defines how the package is built and installed:

```python
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
```

### 3. Robot Description (URDF)

The URDF file describes the robot's physical structure:

```xml
<?xml version="1.0"?>
<robot name="humanoid_skeleton">

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Torso -->
  <link name="torso">
    <visual>
      <geometry>
        <box size="0.2 0.1 0.4"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.1 0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.2" ixy="0.0" ixz="0.0" iyy="0.2" iyz="0.0" izz="0.2"/>
    </inertial>
  </link>

  <joint name="base_to_torso" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.25"/>
  </joint>

  <!-- Head -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="torso_to_head" type="fixed">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.3"/>
  </joint>

  <!-- Left Arm -->
  <link name="left_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
      <material name="green">
        <color rgba="0 1 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="left_shoulder" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.15 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <link name="left_lower_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.04"/>
      </geometry>
      <material name="green">
        <color rgba="0 1 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.3"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="left_elbow" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_lower_arm"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <!-- Right Arm -->
  <link name="right_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
      <material name="green">
        <color rgba="0 1 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="right_shoulder" type="revolute">
    <parent link="torso"/>
    <child link="right_upper_arm"/>
    <origin xyz="-0.15 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <link name="right_lower_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.04"/>
      </geometry>
      <material name="green">
        <color rgba="0 1 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.3"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="right_elbow" type="revolute">
    <parent link="right_upper_arm"/>
    <child link="right_lower_arm"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <!-- Left Leg -->
  <link name="left_upper_leg">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
      <material name="purple">
        <color rgba="0.5 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>
  </link>

  <joint name="left_hip" type="revolute">
    <parent link="base_link"/>
    <child link="left_upper_leg"/>
    <origin xyz="0.08 0 -0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <link name="left_lower_leg">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
      <material name="purple">
        <color rgba="0.5 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.6"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>
  </link>

  <joint name="left_knee" type="revolute">
    <parent link="left_upper_leg"/>
    <child link="left_lower_leg"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <!-- Right Leg -->
  <link name="right_upper_leg">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
      <material name="purple">
        <color rgba="0.5 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>
  </link>

  <joint name="right_hip" type="revolute">
    <parent link="base_link"/>
    <child link="right_upper_leg"/>
    <origin xyz="-0.08 0 -0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <link name="right_lower_leg">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
      <material name="purple">
        <color rgba="0.5 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.6"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>
  </link>

  <joint name="right_knee" type="revolute">
    <parent link="right_upper_leg"/>
    <child link="right_lower_leg"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

</robot>
```

### 4. Launch File

The launch file coordinates starting multiple nodes together:

```python
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

        # Robot controller node
        Node(
            package='minimal_robot_package',
            executable='robot_controller',
            name='robot_controller',
            output='screen'
        )
    ])
```

## Building and Running the Package

### 1. Workspace Setup

Create a ROS 2 workspace and build the package:

```bash
# Create workspace
mkdir -p ~/ros2_workspace/src
cd ~/ros2_workspace/src

# Copy your package to the src directory
# (minimal_robot_package should be in src/)

# Build the workspace
cd ~/ros2_workspace
colcon build --packages-select minimal_robot_package

# Source the workspace
source install/setup.bash
```

### 2. Running the Package

Run the entire system using the launch file:

```bash
# Launch the entire system
ros2 launch minimal_robot_package minimal_robot.launch.py
```

Or run individual nodes:

```bash
# Run publisher only
ros2 run minimal_robot_package minimal_publisher

# Run subscriber only
ros2 run minimal_robot_package minimal_subscriber

# Run controller only
ros2 run minimal_robot_package robot_controller
```

## Package Best Practices

### 1. Code Organization

- Keep nodes focused on a single responsibility
- Use meaningful names for packages, nodes, and topics
- Follow ROS naming conventions (snake_case for packages, nodes, and topics)

### 2. Configuration Management

- Use parameters for configuration instead of hardcoding values
- Separate configuration files from code
- Use launch files to set parameter values

### 3. Error Handling

- Implement proper error handling and logging
- Use appropriate ROS logging levels (info, warn, error, debug)
- Gracefully handle resource cleanup

### 4. Testing

- Write unit tests for your nodes
- Test communication patterns
- Validate URDF for proper structure

## Debugging Tips

### 1. Checking Topics

```bash
# List all topics
ros2 topic list

# Echo a topic to see messages
ros2 topic echo /topic_name std_msgs/msg/String

# Check topic info
ros2 topic info /topic_name
```

### 2. Checking Nodes

```bash
# List all nodes
ros2 node list

# Check node info
ros2 node info /node_name
```

### 3. Checking Parameters

```bash
# List parameters for a node
ros2 param list /node_name

# Get parameter value
ros2 param get /node_name parameter_name
```

## Advanced Topics

### 1. Adding Sensors to the Robot

To add sensors to your robot, extend the URDF:

```xml
<!-- Add a laser scanner to the head -->
<link name="laser_scanner">
  <visual>
    <geometry>
      <cylinder radius="0.02" length="0.05"/>
    </geometry>
    <material name="black">
      <color rgba="0 0 0 1"/>
    </material>
  </visual>
  <collision>
    <geometry>
      <cylinder radius="0.02" length="0.05"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="0.1"/>
    <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
  </inertial>
</link>

<joint name="head_to_laser" type="fixed">
  <parent link="head"/>
  <child link="laser_scanner"/>
  <origin xyz="0.05 0 0"/>
</joint>

<!-- Add Gazebo plugin for the laser scanner -->
<gazebo reference="laser_scanner">
  <sensor type="ray" name="laser_scanner_sensor">
    <pose>0 0 0 0 0 0</pose>
    <visualize>true</visualize>
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>360</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>10.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="laser_scanner_controller" filename="libgazebo_ros_ray_sensor.so">
      <ros>
        <namespace>/robot</namespace>
        <remapping>~/out:=scan</remapping>
      </ros>
      <output_type>sensor_msgs/LaserScan</output_type>
    </plugin>
  </sensor>
</gazebo>
```

### 2. Adding Controllers

For more complex robot control, you can use the ROS 2 Control framework:

```xml
<!-- Add transmission for joint control -->
<transmission name="left_shoulder_trans">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="left_shoulder">
    <hardwareInterface>position_controllers/JointPositionController</hardwareInterface>
  </joint>
  <actuator name="left_shoulder_motor">
    <hardwareInterface>position_controllers/JointPositionController</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

## Summary

Creating a complete ROS 2 robot package involves understanding the structure, organizing files properly, and implementing all the communication patterns learned in this module. The minimal robot package serves as a template that can be extended for more complex robotic applications.

## Learning Check

After completing this section, you should be able to:
- Create a complete ROS 2 package with proper structure
- Define robot models using URDF
- Implement multiple nodes with different communication patterns
- Create launch files to coordinate system startup
- Apply best practices for package development
- Debug common issues in ROS 2 packages