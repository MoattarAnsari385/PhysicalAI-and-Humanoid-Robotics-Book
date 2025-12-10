---
title: "ROS 2 Architecture and Communication Patterns"
sidebar_position: 2
description: "Deep dive into ROS 2 architecture and advanced communication patterns including actions"
---

# ROS 2 Architecture and Communication Patterns

## Architecture Overview

ROS 2 is built on a client-server model that uses the Data Distribution Service (DDS) as its middleware. This architecture provides improved real-time performance, reliability, and security compared to ROS 1. Unlike ROS 1's centralized master architecture, ROS 2 uses a distributed architecture where nodes discover each other automatically.

## DDS Middleware

The Data Distribution Service (DDS) is a specification that defines a standard for distributed, real-time data exchange. In ROS 2, DDS provides:

- **Discovery**: Automatic discovery of nodes and topics
- **Communication**: Reliable message delivery with configurable QoS
- **Transport**: Support for multiple transport protocols (UDP, TCP, shared memory)
- **Quality of Service (QoS)**: Configurable reliability, durability, and performance settings

### QoS Profiles

Quality of Service profiles allow you to configure how messages are delivered:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Example: Reliable communication with keep-all history
qos_profile = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_ALL
)
```

## Actions: Advanced Communication Pattern

Actions are a communication pattern for long-running tasks that provide feedback during execution. They combine the benefits of services (request-response) and topics (streaming data) with additional features:

- **Goal**: Request to perform a long-running task
- **Feedback**: Streaming updates during task execution
- **Result**: Final outcome of the task

### Action Example

```python
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):

    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback)

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])

            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f'Publishing feedback: {feedback_msg.sequence}')

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        return result
```

## Parameters: Configuration Management

Parameters in ROS 2 provide a way to configure nodes at runtime. They can be set at launch time, through command line, or programmatically during execution.

```python
# Declaring and using parameters
class ParameterNode(Node):

    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('robot_name', 'default_robot')
        self.declare_parameter('max_velocity', 1.0)

        # Get parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.max_velocity = self.get_parameter('max_velocity').value
```

## Lifecycle Nodes

Lifecycle nodes provide a way to manage the state of nodes through a well-defined state machine. This is particularly useful for complex nodes that need initialization, activation, and deactivation steps.

## Client Library Support

ROS 2 supports multiple client libraries:
- **rclcpp**: C++ client library
- **rclpy**: Python client library
- **rcljava**: Java client library
- **rclnodejs**: Node.js client library

## Communication Patterns Summary

| Pattern | Type | Use Case | Characteristics |
|---------|------|----------|-----------------|
| Topics | Publish/Subscribe | Streaming data | Asynchronous, multiple publishers/subscribers |
| Services | Request/Response | Simple queries | Synchronous, blocking calls |
| Actions | Goal/Feedback/Result | Long-running tasks | Asynchronous with feedback |
| Parameters | Configuration | Runtime configuration | Dynamic value changes |

## Security Features

ROS 2 includes security features that were not available in ROS 1:
- **Authentication**: Verify node identity
- **Authorization**: Control what nodes can do
- **Encryption**: Encrypt data in transit

## Best Practices for Architecture

1. **Modularity**: Design nodes to be independent and focused
2. **QoS Matching**: Ensure QoS profiles match between publishers and subscribers
3. **Resource Management**: Properly manage memory and computational resources
4. **Error Handling**: Implement robust error handling and recovery
5. **Testing**: Write tests for each node and communication pattern

## Summary

ROS 2's architecture provides a robust foundation for building distributed robotic systems. The DDS middleware, combined with multiple communication patterns, allows for flexible and reliable robot software development. Understanding these architectural concepts is essential for designing scalable and maintainable robotic applications.

## Learning Check

After completing this section, you should be able to:
- Explain the DDS middleware and its role in ROS 2
- Configure Quality of Service settings for different requirements
- Implement and use actions for long-running tasks
- Manage parameters for runtime configuration
- Choose appropriate communication patterns for different use cases