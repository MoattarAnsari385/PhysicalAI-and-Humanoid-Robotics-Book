---
title: "Understanding Nodes, Topics, and Services"
sidebar_position: 1
description: "Learn the fundamental communication patterns in ROS 2: nodes, topics, and services"
---

# Understanding Nodes, Topics, and Services

## Introduction to ROS 2 Architecture

The Robot Operating System 2 (ROS 2) provides a flexible framework for writing robot software. At its core, ROS 2 is designed around a distributed computing model where multiple processes, called nodes, communicate with each other through a publish-subscribe messaging system.

## Nodes: The Building Blocks of ROS 2

A node is a process that performs computation in ROS. Nodes are combined together into a graph and communicate with each other using topics, services, actions, and parameters. In ROS 2, nodes are designed to be modular and reusable components that can be combined to create complex robotic systems.

### Creating a Node

In ROS 2, nodes are typically implemented in one of the supported languages (C++ or Python). Here's a basic example of a node in Python:

```python
import rclpy
from rclpy.node import Node

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

This example demonstrates the basic structure of a ROS 2 node that publishes messages to a topic.

## Topics: Publish-Subscribe Communication

Topics in ROS 2 provide a way for nodes to exchange messages through a publish-subscribe communication pattern. This is an asynchronous communication method where publishers send messages to a topic without knowing which subscribers will receive them.

### Key Characteristics of Topics:
- Asynchronous communication
- Multiple publishers and subscribers can connect to the same topic
- Messages are distributed to all subscribers
- No direct connection between publishers and subscribers

## Services: Request-Response Communication

Services provide a synchronous request-response communication pattern between nodes. When a client sends a request to a service, it waits for a response before continuing execution.

### Service Architecture:
- Service Server: Implements the service and responds to requests
- Service Client: Sends requests to the service and waits for responses

## Practical Example: Simple Communication

Let's look at a practical example that demonstrates both topics and services working together in a simple robot control scenario.

### Topic Example: Robot Position Updates

```python
# Publisher example
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point

class RobotPositionPublisher(Node):

    def __init__(self):
        super().__init__('robot_position_publisher')
        self.publisher = self.create_publisher(Point, 'robot_position', 10)

    def publish_position(self, x, y, z):
        position_msg = Point()
        position_msg.x = x
        position_msg.y = y
        position_msg.z = z
        self.publisher.publish(position_msg)
        self.get_logger().info(f'Published position: ({x}, {y}, {z})')
```

### Service Example: Robot Movement Commands

```python
# Service server example
import rclpy
from rclpy.node import Node
from example_interfaces.srv import Trigger

class RobotMovementServer(Node):

    def __init__(self):
        super().__init__('robot_movement_server')
        self.srv = self.create_service(Trigger, 'move_robot', self.move_robot_callback)

    def move_robot_callback(self, request, response):
        # Implement robot movement logic here
        self.get_logger().info('Moving robot...')
        response.success = True
        response.message = 'Robot moved successfully'
        return response
```

## Best Practices

1. **Node Design**: Keep nodes focused on a single responsibility
2. **Topic Naming**: Use descriptive, consistent names for topics
3. **Message Types**: Choose appropriate message types for your data
4. **QoS Settings**: Consider Quality of Service settings for real-time requirements
5. **Error Handling**: Implement proper error handling and logging

## Summary

Nodes, topics, and services form the foundation of ROS 2 communication. Understanding these concepts is crucial for building distributed robotic systems that can scale and maintain reliability. As you progress through this module, you'll see how these basic building blocks combine to create more sophisticated robotic applications.

## Learning Check

After completing this section, you should be able to:
- Explain the role of nodes in ROS 2 architecture
- Implement basic publisher and subscriber nodes
- Create and use services for request-response communication
- Understand the differences between topics and services