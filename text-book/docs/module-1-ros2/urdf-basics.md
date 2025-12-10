---
title: "Understanding URDF (Unified Robot Description Format) for Humanoids"
sidebar_position: 4
description: "Learn how to model humanoid robots using URDF (Unified Robot Description Format) for ROS 2"
---

# Understanding URDF (Unified Robot Description Format) for Humanoids

This section covers the Unified Robot Description Format (URDF) specifically for humanoid robots. URDF is an XML format for representing robot models that's essential for humanoid robotics, allowing you to define the complex kinematic structure of bipedal robots with multiple degrees of freedom.

## Introduction to URDF for Humanoid Robots

URDF (Unified Robot Description Format) is the standard format for describing robot models in ROS. For humanoid robots, URDF is crucial for defining the complex structure including torso, head, arms, and legs with appropriate joints for locomotion and manipulation.

### Basic Humanoid URDF Structure

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Base link (torso) -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

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
      <mass value="2"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Joint connecting head to torso -->
  <joint name="neck_joint" type="revolute">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0 0 0.35"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="100" velocity="1"/>
  </joint>
</robot>
```

## Humanoid Joint Types and Ranges

Humanoid robots require specific joint types and ranges to enable natural movement:

### Hip Joints (6 DOF for each leg)
```xml
<joint name="left_hip_yaw" type="revolute">
  <parent link="base_link"/>
  <child link="left_thigh"/>
  <origin xyz="0 0.1 -0.1"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.0" upper="1.0" effort="200" velocity="2"/>
  <dynamics damping="1.0" friction="0.1"/>
</joint>

<joint name="left_hip_roll" type="revolute">
  <parent link="left_thigh"/>
  <child link="left_thigh_upper"/>
  <origin xyz="0 0 -0.1"/>
  <axis xyz="1 0 0"/>
  <limit lower="-0.5" upper="0.5" effort="200" velocity="2"/>
</joint>

<joint name="left_hip_pitch" type="revolute">
  <parent link="left_thigh_upper"/>
  <child link="left_shin"/>
  <origin xyz="0 0 -0.3"/>
  <axis xyz="0 1 0"/>
  <limit lower="-2.0" upper="0.5" effort="200" velocity="2"/>
</joint>
```

## Complete Humanoid Example

Here's a more complete humanoid model with arms and legs:

```xml
<?xml version="1.0"?>
<robot name="complete_humanoid">
  <!-- Torso -->
  <link name="torso">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.6"/>
      </geometry>
      <material name="light_grey">
        <color rgba="0.7 0.7 0.7 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.6"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="15"/>
      <inertia ixx="1.5" ixy="0.0" ixz="0.0" iyy="1.5" iyz="0.0" izz="0.5"/>
    </inertial>
  </link>

  <!-- Head -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.12"/>
      </geometry>
      <material name="skin">
        <color rgba="0.9 0.8 0.7 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.12"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="3"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>
  </link>

  <!-- Neck joint -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.35"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="50" velocity="2"/>
  </joint>

  <!-- Left Arm -->
  <link name="left_shoulder">
    <visual>
      <geometry>
        <box size="0.15 0.1 0.1"/>
      </geometry>
      <material name="arm_color">
        <color rgba="0.5 0.5 1 1"/>
      </material>
    </visual>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_shoulder"/>
    <origin xyz="0.15 0.1 0.2"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="2"/>
  </joint>

  <!-- Right Arm (similar structure) -->
  <link name="right_shoulder">
    <visual>
      <geometry>
        <box size="0.15 0.1 0.1"/>
      </geometry>
      <material name="arm_color"/>
    </visual>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="right_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="right_shoulder"/>
    <origin xyz="0.15 -0.1 0.2"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="2"/>
  </joint>

  <!-- Left Leg -->
  <link name="left_hip">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.15"/>
      </geometry>
      <material name="leg_color">
        <color rgba="0.2 0.2 0.8 1"/>
      </material>
    </visual>
    <inertial>
      <mass value="2"/>
      <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.005"/>
    </inertial>
  </link>

  <joint name="left_hip_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_hip"/>
    <origin xyz="-0.1 0.1 -0.1"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="200" velocity="2"/>
  </joint>

  <!-- Right Leg -->
  <link name="right_hip">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.15"/>
      </geometry>
      <material name="leg_color"/>
    </visual>
    <inertial>
      <mass value="2"/>
      <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.005"/>
    </inertial>
  </link>

  <joint name="right_hip_joint" type="revolute">
    <parent link="torso"/>
    <child link="right_hip"/>
    <origin xyz="-0.1 -0.1 -0.1"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="200" velocity="2"/>
  </joint>
</robot>
```

## URDF with Gazebo Integration

For simulation in Gazebo, you can add Gazebo-specific tags:

```xml
<gazebo reference="torso">
  <material>Gazebo/Grey</material>
  <mu1>0.2</mu1>
  <mu2>0.2</mu2>
  <self_collide>false</self_collide>
</gazebo>

<gazebo>
  <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
    <robotNamespace>/humanoid</robotNamespace>
  </plugin>
</gazebo>
```

## Xacro for Complex Humanoid Models

For complex humanoid models, Xacro (XML Macros) helps manage complexity:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_xacro">
  <xacro:property name="M_PI" value="3.1415926535897931" />

  <!-- Define a macro for limbs -->
  <xacro:macro name="limb" params="side prefix xyz">
    <link name="${prefix}_${side}_upper">
      <visual>
        <geometry>
          <box size="0.08 0.08 0.3"/>
        </geometry>
        <material name="limb_color">
          <color rgba="0.5 0.5 0.5 1"/>
        </material>
      </visual>
      <collision>
        <geometry>
          <box size="0.08 0.08 0.3"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="2"/>
        <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.001"/>
      </inertial>
    </link>

    <joint name="${prefix}_${side}_joint" type="revolute">
      <parent link="torso"/>
      <child link="${prefix}_${side}_upper"/>
      <origin xyz="${xyz}"/>
      <axis xyz="0 1 0"/>
      <limit lower="-2.0" upper="2.0" effort="100" velocity="2"/>
    </joint>
  </xacro:macro>

  <!-- Use the macro to create limbs -->
  <xacro:limb side="left" prefix="arm" xyz="0.15 0.1 0.2"/>
  <xacro:limb side="right" prefix="arm" xyz="0.15 -0.1 0.2"/>
  <xacro:limb side="left" prefix="leg" xyz="-0.1 0.1 -0.1"/>
  <xacro:limb side="right" prefix="leg" xyz="-0.1 -0.1 -0.1"/>
</robot>
```

## Best Practices for Humanoid URDF

1. **Proper Inertial Properties**: Accurate mass and inertia values are crucial for stable simulation
2. **Joint Limits**: Set realistic joint limits based on human anatomy
3. **Collision Avoidance**: Ensure proper collision geometry to prevent self-collision
4. **Kinematic Chains**: Structure joints in proper kinematic chains (torso → head, torso → arms, torso → legs)
5. **ROS Integration**: Include proper ROS control interfaces for real robot control

## Summary

URDF is fundamental for humanoid robotics, enabling the description of complex multi-degree-of-freedom robots. Proper URDF modeling is essential for both simulation and real-world control of humanoid robots, providing the kinematic structure needed for motion planning, control, and perception algorithms.