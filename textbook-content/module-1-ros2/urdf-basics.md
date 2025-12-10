---
title: "URDF Basics and Robot Modeling"
sidebar_position: 3
description: "Learn the fundamentals of Unified Robot Description Format (URDF) for robot modeling"
---

# URDF Basics and Robot Modeling

## Introduction to URDF

The Unified Robot Description Format (URDF) is an XML format for representing a robot model. It defines the physical and visual properties of a robot, including its links, joints, and other components. URDF is essential for simulation, visualization, and control of robots in ROS.

## URDF Structure

A URDF file describes a robot as a collection of links connected by joints. The structure follows this pattern:

- **Links**: Rigid parts of the robot (e.g., chassis, arms, wheels)
- **Joints**: Connections between links that allow relative motion
- **Visual**: How the link appears in visualization tools
- **Collision**: How the link interacts with the environment in simulation
- **Inertial**: Physical properties for simulation (mass, center of mass, inertia)

## Basic URDF Elements

### Links

A link represents a rigid body in the robot. Each link can have visual, collision, and inertial properties:

```xml
<link name="link_name">
  <visual>
    <geometry>
      <box size="0.1 0.1 0.1"/>
    </geometry>
    <material name="color">
      <color rgba="1 0 0 1"/>
    </material>
  </visual>
  <collision>
    <geometry>
      <box size="0.1 0.1 0.1"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="1.0"/>
    <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
  </inertial>
</link>
```

### Joints

Joints connect links and define how they can move relative to each other:

```xml
<joint name="joint_name" type="revolute">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
</joint>
```

Joint types include:
- **Fixed**: No movement allowed
- **Revolute**: Rotational movement around an axis
- **Continuous**: Rotational movement without limits
- **Prismatic**: Linear movement along an axis
- **Floating**: 6 degrees of freedom
- **Planar**: Movement in a plane

## Geometry Types

URDF supports several geometry types:

- **Box**: Defined by size="x y z"
- **Cylinder**: Defined by radius and length
- **Sphere**: Defined by radius
- **Mesh**: Defined by filename and scale

## Materials and Colors

Materials define the visual appearance of links:

```xml
<material name="red">
  <color rgba="1 0 0 1"/>
</material>
```

Colors use RGBA values where each component ranges from 0.0 to 1.0.

## Robot State Publisher

To use URDF in ROS, the robot_state_publisher node publishes the robot's joint states and transforms:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
import math

class RobotStatePublisher(Node):
    def __init__(self):
        super().__init__('robot_state_publisher')
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.timer = self.create_timer(0.1, self.publish_joint_states)

    def publish_joint_states(self):
        msg = JointState()
        msg.name = ['joint1', 'joint2']
        msg.position = [math.sin(self.get_clock().now().nanoseconds * 1e-9),
                        math.cos(self.get_clock().now().nanoseconds * 1e-9)]
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        self.joint_pub.publish(msg)
```

## Practical Example: Simple Arm

Here's a complete example of a simple 2-DOF arm:

```xml
<?xml version="1.0"?>
<robot name="simple_arm">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Upper arm -->
  <link name="upper_arm">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Elbow joint -->
  <joint name="shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="upper_arm"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <!-- Lower arm -->
  <link name="lower_arm">
    <visual>
      <geometry>
        <cylinder radius="0.04" length="0.25"/>
      </geometry>
      <material name="green">
        <color rgba="0 1 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.04" length="0.25"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.3"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Wrist joint -->
  <joint name="elbow_joint" type="revolute">
    <parent link="upper_arm"/>
    <child link="lower_arm"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>
</robot>
```

## Best Practices

1. **Use consistent naming**: Follow a consistent naming convention for links and joints
2. **Proper mass properties**: Accurate inertial properties are crucial for simulation
3. **Collision vs visual**: Use simple shapes for collision and detailed shapes for visual
4. **Joint limits**: Always specify appropriate joint limits to prevent damage
5. **Xacro for complex robots**: Use Xacro (XML Macros) for complex robots to avoid repetition

## Xacro Introduction

For complex robots, Xacro allows you to define macros and reuse components:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="simple_arm_xacro">
  <xacro:property name="PI" value="3.14159"/>

  <xacro:macro name="cylinder_inertia" params="m r h">
    <inertia ixx="${m*(3*r*r+h*h)/12}" ixy="0" ixz="0"
             iyy="${m*(3*r*r+h*h)/12}" iyz="0"
             izz="${m*r*r/2}"/>
  </xacro:macro>

  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <xacro:cylinder_inertia m="1.0" r="0.1" h="0.1"/>
      <origin xyz="0 0 0"/>
    </inertial>
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.1"/>
      </geometry>
    </collision>
  </link>
</robot>
```

## Visualization with RViz

To visualize your URDF in RViz:
1. Launch robot_state_publisher with your URDF
2. Add RobotModel display in RViz
3. Set the TF frame to your robot's base link

## Troubleshooting Common Issues

- **Missing transforms**: Ensure all joints are properly connected
- **Inverted axes**: Check joint axis orientation
- **Simulation instability**: Verify mass and inertia properties
- **Visualization issues**: Check material definitions and file paths

## Summary

URDF is fundamental to robot modeling in ROS. It provides a standardized way to describe robot geometry, kinematics, and dynamics. Understanding URDF is essential for simulation, visualization, and control of robotic systems.

## Learning Check

After completing this section, you should be able to:
- Create basic URDF files with links and joints
- Define visual and collision properties for robot components
- Use different joint types appropriately
- Understand the relationship between URDF and robot simulation
- Apply best practices for URDF modeling