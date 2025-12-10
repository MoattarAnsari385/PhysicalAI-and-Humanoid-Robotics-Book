---
title: "Bridging Python Agents to ROS Controllers using rclpy"
sidebar_position: 2
description: "Learn how to bridge Python agents to ROS controllers using rclpy client library for humanoid robotics"
---

# Bridging Python Agents to ROS Controllers using rclpy

This section focuses on how to bridge Python-based AI agents to ROS 2 controllers using the rclpy client library. This is essential for humanoid robotics where high-level AI decision-making in Python needs to interface with low-level robot control systems in ROS 2.

## Introduction to rclpy for AI Integration

rclpy is the Python client library for ROS 2 that enables Python agents to communicate with ROS 2 systems. For humanoid robotics, this allows AI algorithms (written in Python with libraries like TensorFlow, PyTorch, or OpenAI Gym) to control robot hardware through ROS 2.

## Basic Bridge Pattern

The fundamental pattern for bridging Python agents to ROS controllers:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import numpy as np

class PythonAgentBridge(Node):
    def __init__(self):
        super().__init__('python_agent_bridge')

        # Subscribers for sensor data from robot
        self.joint_state_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        # Publishers for control commands to robot
        self.control_pub = self.create_publisher(
            Float64MultiArray,
            'joint_commands',
            10
        )

        # Timer for agent control loop
        self.control_timer = self.create_timer(0.05, self.agent_control_loop)  # 20Hz

        # Internal state for agent
        self.current_joint_states = None
        self.agent_action = None

    def joint_state_callback(self, msg):
        """Receive joint states from robot"""
        self.current_joint_states = np.array(msg.position)
        self.get_logger().debug(f'Received joint states: {self.current_joint_states}')

    def agent_control_loop(self):
        """Main control loop that interfaces Python agent with ROS"""
        if self.current_joint_states is not None:
            # Process with Python agent
            action = self.python_agent_policy(self.current_joint_states)

            # Send action to robot via ROS
            self.send_action_to_robot(action)

    def python_agent_policy(self, joint_states):
        """Placeholder for Python-based AI policy"""
        # This could be a neural network, rule-based system, etc.
        # For example: return a simple PD controller action
        target_positions = joint_states + 0.1  # Simple example
        return target_positions

    def send_action_to_robot(self, action):
        """Send control action to robot"""
        cmd_msg = Float64MultiArray()
        cmd_msg.data = action.tolist()
        self.control_pub.publish(cmd_msg)
```

## Advanced: Reinforcement Learning Agent Integration

Example of integrating a reinforcement learning agent with humanoid robot control:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Imu
import torch
import torch.nn as nn
import numpy as np

class RLAgentBridge(Node):
    def __init__(self):
        super().__init__('rl_agent_bridge')

        # Robot interface
        self.joint_sub = self.create_subscription(JointState, 'joint_states', self.joint_callback, 10)
        self.imu_sub = self.create_subscription(Imu, 'imu/data', self.imu_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # RL components
        self.rl_policy = self.load_rl_policy()
        self.robot_state = None
        self.control_timer = self.create_timer(0.1, self.rl_control_loop)  # 10Hz

    def load_rl_policy(self):
        """Load pre-trained reinforcement learning policy"""
        # In practice, this would load a trained neural network
        class SimplePolicy(nn.Module):
            def __init__(self, input_size, output_size):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_size, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, output_size)
                )

            def forward(self, x):
                return self.network(x)

        # Initialize with random weights (in practice, load trained weights)
        return SimplePolicy(input_size=10, output_size=2)  # Example sizes

    def joint_callback(self, msg):
        """Process joint state data"""
        self.robot_state = {
            'positions': np.array(msg.position),
            'velocities': np.array(msg.velocity),
            'effort': np.array(msg.effort)
        }

    def imu_callback(self, msg):
        """Process IMU data for balance control"""
        if self.robot_state:
            self.robot_state['imu'] = {
                'orientation': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
                'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
                'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
            }

    def rl_control_loop(self):
        """Main RL control loop"""
        if self.robot_state:
            # Prepare state for RL agent
            state_vector = self.prepare_state_vector(self.robot_state)

            # Get action from RL policy
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
                action_tensor = self.rl_policy(state_tensor)
                action = action_tensor.numpy().squeeze()

            # Execute action on robot
            self.execute_action(action)

    def prepare_state_vector(self, robot_state):
        """Convert robot state to format expected by RL agent"""
        # Combine joint positions, velocities, and IMU data
        positions = robot_state['positions']
        velocities = robot_state['velocities']
        imu_data = robot_state.get('imu', {'orientation': [0,0,0,1], 'angular_velocity': [0,0,0], 'linear_acceleration': [0,0,0]})

        state = np.concatenate([
            positions[:5],  # First 5 joints (example)
            velocities[:5],  # First 5 joint velocities
            imu_data['angular_velocity'],
            imu_data['linear_acceleration']
        ])

        return state

    def execute_action(self, action):
        """Execute action on robot"""
        cmd_msg = Twist()
        cmd_msg.linear.x = float(action[0])  # Forward/backward
        cmd_msg.angular.z = float(action[1])  # Turn
        self.cmd_pub.publish(cmd_msg)
```

## Multi-Agent Coordination Pattern

For humanoid robots with multiple coordinated controllers:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from builtin_interfaces.msg import Duration

class MultiAgentController(Node):
    def __init__(self):
        super().__init__('multi_agent_controller')

        # Multiple publishers for different subsystems
        self.upper_body_pub = self.create_publisher(Float64MultiArray, 'upper_body_commands', 10)
        self.lower_body_pub = self.create_publisher(Float64MultiArray, 'lower_body_commands', 10)
        self.head_pub = self.create_publisher(Float64MultiArray, 'head_commands', 10)

        # Coordinate multiple Python agents
        self.control_timer = self.create_timer(0.05, self.coordinated_control)

    def coordinated_control(self):
        """Coordinate multiple Python agents for full humanoid control"""
        # Get high-level command (could come from voice, vision, etc.)
        high_level_command = self.get_high_level_command()

        # Coordinate different agents
        upper_body_action = self.upper_body_agent(high_level_command)
        lower_body_action = self.lower_body_agent(high_level_command)
        head_action = self.head_agent(high_level_command)

        # Publish coordinated commands
        self.upper_body_pub.publish(upper_body_action)
        self.lower_body_pub.publish(lower_body_action)
        self.head_pub.publish(head_action)

    def upper_body_agent(self, command):
        """Upper body control agent"""
        # Implementation for arms, torso, etc.
        pass

    def lower_body_agent(self, command):
        """Lower body control agent"""
        # Implementation for legs, balance, etc.
        pass

    def head_agent(self, command):
        """Head control agent"""
        # Implementation for gaze, attention, etc.
        pass
```

## Advanced: Vision-Based Control Bridge

Example of bridging computer vision algorithms to robot control:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class VisionControlBridge(Node):
    def __init__(self):
        super().__init__('vision_control_bridge')

        self.bridge = CvBridge()

        # Subscribe to camera feed
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10)

        # Subscribe to camera info for intrinsic parameters
        self.camera_info_sub = self.create_subscription(
            CameraInfo, 'camera/camera_info', self.camera_info_callback, 10)

        # Publish control commands
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Vision processing timer
        self.vision_timer = self.create_timer(0.1, self.vision_processing_loop)  # 10Hz

        # Internal state
        self.current_image = None
        self.camera_matrix = None
        self.distortion_coeffs = None

    def camera_info_callback(self, msg):
        """Process camera intrinsic parameters"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.d)

    def image_callback(self, msg):
        """Process incoming camera image"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.current_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')

    def vision_processing_loop(self):
        """Main vision processing and control loop"""
        if self.current_image is not None:
            # Process image with Python vision algorithms
            control_action = self.process_vision_and_control(self.current_image)

            # Publish control command
            self.cmd_pub.publish(control_action)

    def process_vision_and_control(self, image):
        """Process image and generate control commands"""
        # Example: detect colored objects and move toward them
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define range for red color (example)
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([170, 50, 50])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)

        mask = mask1 + mask2

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cmd_msg = Twist()

        if contours:
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Get center of contour
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Convert to image center relative coordinates
                img_center_x = image.shape[1] / 2
                horizontal_error = cx - img_center_x

                # Generate control based on object position
                cmd_msg.linear.x = 0.2  # Move forward slowly
                cmd_msg.angular.z = -horizontal_error * 0.001  # Turn toward object
            else:
                cmd_msg.linear.x = 0.0
                cmd_msg.angular.z = 0.0
        else:
            # No object found, stop or search
            cmd_msg.linear.x = 0.0
            cmd_msg.angular.z = 0.1  # Slow turn to search

        return cmd_msg
```

## Safety and Error Handling

Critical safety measures when bridging AI agents to physical robots:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
import numpy as np

class SafeAgentBridge(Node):
    def __init__(self):
        super().__init__('safe_agent_bridge')

        # Robot interface
        self.joint_sub = self.create_subscription(JointState, 'joint_states', self.joint_callback, 10)
        self.cmd_pub = self.create_publisher(Float64MultiArray, 'safe_joint_commands', 10)

        # Safety timer
        self.safety_timer = self.create_timer(0.01, self.safety_check)  # 100Hz safety check

        # Internal state
        self.current_joints = None
        self.last_command_time = self.get_clock().now()
        self.emergency_stop = False

    def joint_callback(self, msg):
        """Monitor joint states for safety"""
        self.current_joints = {
            'position': np.array(msg.position),
            'velocity': np.array(msg.velocity),
            'effort': np.array(msg.effort)
        }

    def safety_check(self):
        """Perform safety checks at high frequency"""
        if self.emergency_stop:
            self.publish_emergency_stop()
            return

        # Check if we've received joint states
        if self.current_joints is None:
            self.get_logger().warn('No joint states received - stopping')
            self.publish_emergency_stop()
            return

        # Check for dangerous joint positions
        pos_limits = np.array([2.0] * len(self.current_joints['position']))  # Example limits
        if np.any(np.abs(self.current_joints['position']) > pos_limits):
            self.get_logger().error('Joint position limit exceeded - emergency stop')
            self.emergency_stop = True
            self.publish_emergency_stop()
            return

        # Check for dangerous velocities
        vel_limits = np.array([5.0] * len(self.current_joints['velocity']))  # Example limits
        if np.any(np.abs(self.current_joints['velocity']) > vel_limits):
            self.get_logger().error('Joint velocity limit exceeded - emergency stop')
            self.emergency_stop = True
            self.publish_emergency_stop()
            return

        # Check for timeout (no commands received for too long)
        current_time = self.get_clock().now()
        if (current_time - self.last_command_time).nanoseconds > 1e9:  # 1 second
            self.get_logger().warn('Command timeout - stopping robot')
            self.publish_emergency_stop()
            return

    def publish_emergency_stop(self):
        """Publish zero commands to stop robot"""
        zero_cmd = Float64MultiArray()
        zero_cmd.data = [0.0] * len(self.current_joints['position']) if self.current_joints else [0.0] * 6
        self.cmd_pub.publish(zero_cmd)
```

## Best Practices for Agent Integration

1. **State Synchronization**: Ensure agent and robot states are properly synchronized
2. **Timing Considerations**: Match agent update rates with robot control capabilities
3. **Safety Checks**: Implement safety layers between AI agents and robot hardware
4. **Error Handling**: Robust error handling for communication failures
5. **Logging**: Comprehensive logging for debugging agent-robot interactions

## Summary

Bridging Python agents to ROS controllers enables powerful AI-driven control of humanoid robots. The rclpy library provides the essential interface for connecting high-level AI algorithms with low-level robot control systems, enabling sophisticated autonomous behaviors.