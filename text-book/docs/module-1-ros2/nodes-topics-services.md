---
title: "ROS 2 Nodes, Topics, and Services"
sidebar_position: 1
description: "Learn the fundamental communication patterns in ROS 2 for humanoid robotics: nodes, topics, and services"
---

# ROS 2 Nodes, Topics, and Services

This section covers the fundamental communication patterns in ROS 2: nodes, topics, and services. These are the building blocks of ROS communication, enabling distributed computation and modular design. In the context of humanoid robotics, these patterns allow different subsystems (locomotion, perception, manipulation) to communicate seamlessly.

## Understanding Nodes

Nodes are the fundamental building blocks of ROS 2 applications. A node is a process that performs computation and can publish or subscribe to messages. In humanoid robotics, you might have nodes for:

- Joint controllers
- Sensor processing
- High-level planning
- Perception systems
- Behavior managers

### Creating a Basic Node

```python
import rclpy
from rclpy.node import Node

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')
        self.get_logger().info('Humanoid Controller Node Started')

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Topics: Publish-Subscribe Communication

Topics enable asynchronous communication between nodes using a publish-subscribe pattern. This is ideal for continuous data streams like sensor readings, joint states, or camera feeds.

### Topic Communication Example

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')
        self.publisher = self.create_publisher(JointState, 'joint_states', 10)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = JointState()
        msg.name = ['hip_joint', 'knee_joint', 'ankle_joint']
        msg.position = [0.0, 0.0, 0.0]  # Example positions
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing joint states: {msg.position}')
```

## Services: Request-Response Communication

Services provide synchronous request-response communication, useful for actions that require confirmation or return specific results.

### Service Example

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import SetBool

class JointCalibrationService(Node):
    def __init__(self):
        super().__init__('joint_calibration_service')
        self.srv = self.create_service(
            SetBool, 'calibrate_joints', self.calibrate_joints_callback)

    def calibrate_joints_callback(self, request, response):
        if request.data:  # True means calibrate
            self.get_logger().info('Starting joint calibration...')
            # Perform calibration logic
            response.success = True
            response.message = 'Joint calibration completed'
        else:
            response.success = False
            response.message = 'Calibration cancelled'

        return response
```

## Practical Applications in Humanoid Robotics

In humanoid robotics, these communication patterns work together:

- **Topics** handle continuous sensor data and joint state updates
- **Services** manage discrete operations like calibration or mode switching
- **Nodes** represent modular subsystems that can be developed and tested independently

This architecture enables robust, scalable humanoid robot systems where components can be developed, tested, and maintained independently while maintaining seamless communication.

## Quality of Service (QoS) in Humanoid Robotics

For humanoid robots, QoS settings are critical for ensuring real-time performance:

```python
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# For critical control messages
control_qos = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST
)

# For sensor data where latest values are most important
sensor_qos = QoSProfile(
    depth=5,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST
)
```

## Advanced: Actions for Long-Running Tasks

For complex humanoid behaviors that take time to complete, ROS 2 provides Actions:

```python
from rclpy.action import ActionServer
from example_interfaces.action import Fibonacci

class HumanoidActionServer:
    def __init__(self, node):
        self._action_server = ActionServer(
            node,
            Fibonacci,
            'move_humanoid',
            self.execute_callback)

    def execute_callback(self, goal_handle):
        # Execute complex humanoid movement
        feedback_msg = Fibonacci.Feedback()
        result = Fibonacci.Result()

        # Implementation would include movement execution
        goal_handle.succeed()
        return result
```

## Summary

ROS 2 communication patterns provide the foundation for building complex humanoid robot systems. The combination of nodes, topics, services, and actions enables modular, distributed control architectures that are essential for managing the complexity of humanoid robots with multiple degrees of freedom, sensors, and behaviors.