---
title: "Gazebo Worlds and Environment Setup"
sidebar_position: 1
description: "Learn to create and configure Gazebo simulation worlds with proper physics and environment properties"
---

# Gazebo Worlds and Environment Setup

## Introduction to Gazebo Simulation

Gazebo is a robot simulation environment that provides realistic physics, high-quality graphics, and convenient programmatic interfaces. It's widely used in robotics research and development to test algorithms, robot designs, and control strategies before deploying on real hardware. Gazebo simulates rigid body dynamics, sensors, and environmental conditions with high fidelity.

## Gazebo Architecture

Gazebo consists of several key components:

- **Physics Engine**: Handles collision detection and dynamics simulation (ODE, Bullet, Simbody)
- **Sensor System**: Simulates various sensor types (cameras, LiDAR, IMU, etc.)
- **Rendering Engine**: Provides visual feedback and user interface
- **Communication Interface**: Uses Gazebo Transport for inter-process communication
- **Plugin System**: Allows custom functionality through plugins

## World File Structure

Gazebo worlds are defined using SDF (Simulation Description Format), an XML-based format. A basic world file structure looks like this:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="default">
    <!-- World properties -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Environment properties -->
    <gravity>0 0 -9.8</gravity>

    <!-- Plugins -->
    <plugin name="ground_truth" filename="libgazebo_ros_p3d.so">
      <alwaysOn>true</alwaysOn>
      <updateRate>30.0</updateRate>
      <bodyName>chassis</bodyName>
      <topicName>ground_truth_odom</topicName>
      <gaussianNoise>0.01</gaussianNoise>
    </plugin>

    <!-- Models -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Custom models would be placed here -->
  </world>
</sdf>
```

## Creating Custom Worlds

### Basic World with Physics Properties

Let's create a simple world file that defines a basic environment:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="simple_world">
    <!-- Physics engine configuration -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Environment settings -->
    <scene>
      <ambient>0.4 0.4 0.4 1</ambient>
      <background>0.7 0.7 0.7 1</background>
      <shadows>true</shadows>
    </scene>

    <!-- Lighting -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Add a simple box obstacle -->
    <model name="box_obstacle">
      <pose>2 0 0.5 0 0 0</pose>
      <static>false</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.2 0.1 1</ambient>
            <diffuse>0.8 0.2 0.1 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.166667</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.166667</iyy>
            <iyz>0</iyz>
            <izz>0.166667</izz>
          </inertia>
        </inertial>
      </link>
    </model>
  </world>
</sdf>
```

## World Properties Configuration

### Physics Configuration

The physics section defines how the simulation behaves:

- **max_step_size**: Maximum time step for the physics engine (smaller = more accurate but slower)
- **real_time_factor**: Target ratio of simulation time to real time (1.0 = real-time)
- **real_time_update_rate**: Update rate in Hz (higher = smoother but more CPU intensive)

### Gravity Settings

Gravity is defined as a 3D vector in m/s². The default Earth gravity is `0 0 -9.8`, but this can be changed for different environments (e.g., Moon, Mars).

### Scene Configuration

The scene section controls visual appearance:
- **ambient**: Ambient lighting color
- **background**: Background color
- **shadows**: Enable/disable shadow rendering

## Advanced World Features

### Adding Terrain

For more complex environments, you can add terrain:

```xml
<model name="uneven_terrain">
  <link name="terrain_link">
    <collision name="collision">
      <geometry>
        <heightmap>
          <uri>file://path/to/heightmap.png</uri>
          <size>100 100 20</size>
          <pos>0 0 0</pos>
        </heightmap>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <heightmap>
          <uri>file://path/to/heightmap.png</uri>
          <size>100 100 20</size>
          <pos>0 0 0</pos>
        </heightmap>
      </geometry>
    </visual>
  </link>
</model>
```

### Dynamic Lighting

You can also add custom lighting to your world:

```xml
<light name="custom_light" type="point">
  <pose>5 5 10 0 0 0</pose>
  <diffuse>1 1 1 1</diffuse>
  <specular>0.5 0.5 0.5 1</specular>
  <attenuation>
    <range>20</range>
    <constant>0.9</constant>
    <linear>0.01</linear>
    <quadratic>0.001</quadratic>
  </attenuation>
  <cast_shadows>true</cast_shadows>
</light>
```

## World Best Practices

### 1. Performance Optimization
- Use appropriate step sizes (0.001s for accurate physics, 0.01s for faster simulation)
- Balance real-time factor based on your computational resources
- Simplify collision geometry when possible

### 2. Realism vs. Performance
- Use detailed models for visual elements
- Simplify collision meshes for better performance
- Consider using static models for environment elements that don't need to move

### 3. Testing Different Scenarios
- Create multiple world files for different testing scenarios
- Use world files to test in various environments (indoor, outdoor, different terrains)
- Document the purpose of each world file

## Creating a Humanoid-Focused World

For our humanoid robot, we'll want to create a world that's appropriate for testing locomotion and navigation:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="humanoid_test_world">
    <!-- Physics configuration -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Scene configuration -->
    <scene>
      <ambient>0.3 0.3 0.3 1</ambient>
      <background>0.5 0.5 0.5 1</background>
      <shadows>true</shadows>
    </scene>

    <!-- Lighting -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Add some obstacles for navigation testing -->
    <model name="wall_1">
      <pose>5 0 1 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.1 5 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.1 5 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Add a ramp for testing locomotion -->
    <model name="ramp">
      <pose>0 3 0 0 0 0.785</pose>  <!-- 45 degree angle -->
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>2 1 0.5</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>2 1 0.5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.6 0.2 1</ambient>
            <diffuse>0.8 0.6 0.2 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

## Launching Worlds with ROS 2

To launch your world with ROS 2, you can create a launch file:

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    world_path = PathJoinSubstitution([
        FindPackageShare('my_robot_gazebo'),
        'worlds',
        'humanoid_test_world.sdf'
    ])

    return LaunchDescription([
        # Launch Gazebo with our world
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                FindPackageShare('gazebo_ros'),
                '/launch',
                '/gazebo.launch.py'
            ]),
            launch_arguments={
                'world': world_path,
                'verbose': 'true'
            }.items()
        )
    ])
```

## Troubleshooting Common Issues

### 1. Slow Simulation
- Reduce the update rate
- Increase the step size (but be careful with accuracy)
- Simplify collision geometry

### 2. Unstable Physics
- Decrease the step size
- Check inertial properties of your models
- Verify mass and center of mass settings

### 3. Visual Artifacts
- Adjust lighting settings
- Check texture and material definitions
- Verify model positioning

## Summary

Creating Gazebo worlds involves understanding SDF format, physics properties, and visual settings. Well-designed worlds provide the foundation for effective robot testing and validation. For humanoid robots, creating appropriate test environments with various obstacles and terrains is crucial for developing robust locomotion and navigation capabilities.

## Learning Check

After completing this section, you should be able to:
- Create basic Gazebo world files using SDF
- Configure physics properties appropriately
- Add static and dynamic objects to your world
- Set up lighting and visual properties
- Understand the balance between realism and performance