---
title: "rclpy Examples and Advanced Patterns"
sidebar_position: 4
description: "Advanced examples and patterns using the ROS 2 Python client library"
---

# rclpy Examples and Advanced Patterns

## Introduction to rclpy

The ROS Client Library for Python (rclpy) provides the Python API for ROS 2. It allows you to create nodes, publish and subscribe to topics, provide and call services, and work with actions. This section covers advanced patterns and examples beyond basic usage.

## Advanced Node Patterns

### Lifecycle Nodes

Lifecycle nodes provide a way to manage the state of nodes through a well-defined state machine:

```python
import rclpy
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rclpy.lifecycle import Publisher
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String


class LifecyclePublisher(LifecycleNode):

    def __init__(self, node_name):
        super().__init__(node_name)
        self.pub = None

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.pub = self.create_lifecycle_publisher(
            String, 'lifecycle_chatter', qos_profile=qos_profile_sensor_data)
        self.get_logger().info('Lifecycle publisher is configured')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.pub.on_activate()
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.get_logger().info('Lifecycle publisher is activated')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.timer.cancel()
        self.pub.on_deactivate()
        self.get_logger().info('Lifecycle publisher is deactivated')
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.destroy_timer(self.timer)
        self.destroy_publisher(self.pub)
        self.get_logger().info('Lifecycle publisher is cleaned up')
        return TransitionCallbackReturn.SUCCESS

    def timer_callback(self):
        msg = String()
        msg.data = 'Lifecycle msg: %d' % self.get_clock().now().nanoseconds
        self.get_logger().info('Lifecycle publisher: Publishing: "%s"' % msg.data)
        self.pub.publish(msg)
```

### Composition and Components

ROS 2 supports node composition where multiple nodes can run in the same process:

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import String


class SimpleComponent(Node):
    def __init__(self):
        super().__init__('simple_component')
        self.publisher = self.create_publisher(String, 'topic', QoSProfile(depth=10))
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello from component'
        self.publisher.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)


def main(args=None):
    rclpy.init(args=args)

    # Create a container node for composition
    container = rclpy.create_composable_node_container(
        node_name='container',
        namespace='',
        package_name='rclpy',
        plugin_name='composition::Talker')

    # Add components to container
    component = SimpleComponent()
    container.add_node(component)

    try:
        rclpy.spin(container)
    except KeyboardInterrupt:
        pass
    finally:
        container.destroy_node()
        rclpy.shutdown()
```

## Advanced Communication Patterns

### Quality of Service (QoS) Settings

QoS profiles allow you to configure how messages are delivered:

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import String


class QoSPublisher(Node):
    def __init__(self):
        super().__init__('qos_publisher')

        # Reliable communication with keep-all history
        reliable_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_ALL,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # Best-effort communication with limited history
        best_effort_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE
        )

        self.reliable_publisher = self.create_publisher(String, 'reliable_topic', reliable_qos)
        self.best_effort_publisher = self.create_publisher(String, 'best_effort_topic', best_effort_qos)

        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = 'QoS message'

        self.reliable_publisher.publish(msg)
        self.best_effort_publisher.publish(msg)
        self.get_logger().info('Published messages with different QoS settings')
```

### Parameter Handling

Advanced parameter handling with callbacks:

```python
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import ParameterDescriptor


class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with descriptors
        self.declare_parameter(
            'robot_name',
            'default_robot',
            ParameterDescriptor(description='Name of the robot'))

        self.declare_parameter(
            'max_velocity',
            1.0,
            ParameterDescriptor(description='Maximum velocity of the robot'))

        # Set up parameter callback
        self.add_on_set_parameters_callback(self.parameters_callback)

        # Create parameter service
        self.param_service = self.create_service(
            SetParameters,
            'set_parameters',
            self.set_parameters_callback)

    def parameters_callback(self, params):
        """Callback for parameter changes"""
        for param in params:
            if param.name == 'max_velocity' and param.value < 0.0:
                self.get_logger().warn('Max velocity cannot be negative')
                return SetParameters.Result(successful=False, reason='Velocity must be positive')
        return SetParameters.Result(successful=True)

    def set_parameters_callback(self, request, response):
        """Service callback for setting parameters"""
        for param in request.parameters:
            if self.has_parameter(param.name):
                self.set_parameters([param])
        return response
```

## Service and Action Patterns

### Advanced Service Implementation

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts
from threading import Lock


class ThreadSafeService(Node):
    def __init__(self):
        super().__init__('thread_safe_service')
        self.service = self.create_service(
            AddTwoInts,
            'add_two_ints',
            self.add_two_ints_callback)

        # Use lock for thread safety if needed
        self.lock = Lock()
        self.request_count = 0

    def add_two_ints_callback(self, request, response):
        with self.lock:
            self.request_count += 1
            self.get_logger().info(f'Request #{self.request_count}: {request.a} + {request.b}')

        response.sum = request.a + request.b
        return response
```

### Action Client Implementation

```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from example_interfaces.action import Fibonacci


class FibonacciActionClient(Node):

    def __init__(self):
        super().__init__('fibonacci_action_client')
        self._action_client = ActionClient(self, Fibonacci, 'fibonacci')

    def send_goal(self, order):
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        self._action_client.wait_for_server()

        # Send goal and get future
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Received feedback: {feedback.sequence}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.sequence}')
        rclpy.shutdown()
```

## Error Handling and Testing

### Comprehensive Error Handling

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.exceptions import ParameterNotDeclaredException
import traceback


class RobustNode(Node):
    def __init__(self):
        super().__init__('robust_node')
        self.publisher = self.create_publisher(String, 'robust_topic', 10)
        self.timer = self.create_timer(1.0, self.safe_timer_callback)

        # Parameter validation
        try:
            self.declare_parameter('operation_mode', 'normal')
            self.operation_mode = self.get_parameter('operation_mode').value
        except ParameterNotDeclaredException:
            self.get_logger().error('Parameter not declared')
            self.operation_mode = 'normal'

    def safe_timer_callback(self):
        try:
            if self.operation_mode == 'safe':
                self.get_logger().warn('Operating in safe mode')
                return

            msg = String()
            msg.data = 'Robust message'
            self.publisher.publish(msg)

        except Exception as e:
            self.get_logger().error(f'Error in timer callback: {str(e)}')
            self.get_logger().error(f'Traceback: {traceback.format_exc()}')
```

## Async/Await Patterns

Using async/await for non-blocking operations:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import asyncio
from rclpy.executors import SingleThreadedExecutor


class AsyncNode(Node):
    def __init__(self):
        super().__init__('async_node')
        self.publisher = self.create_publisher(String, 'async_topic', 10)

        # Start async task
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self)

        # Run async function
        self.async_task = self.create_timer(1.0, self.async_timer_callback)

    async def async_operation(self):
        """Simulate async operation"""
        await asyncio.sleep(0.5)  # Non-blocking sleep
        return "Async result"

    def async_timer_callback(self):
        # In practice, you'd use rclpy's async capabilities
        msg = String()
        msg.data = 'Async message'
        self.publisher.publish(msg)
        self.get_logger().info('Published async message')
```

## Best Practices

1. **Resource Management**: Always properly clean up resources in destroy methods
2. **Error Handling**: Implement comprehensive error handling with logging
3. **Threading**: Be aware of thread safety when using shared resources
4. **QoS Matching**: Ensure QoS profiles match between publishers and subscribers
5. **Parameter Validation**: Validate parameters at runtime
6. **Testing**: Write unit tests for your nodes and components

## Testing Patterns

```python
import unittest
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from example_interfaces.srv import AddTwoInts


class TestNode(Node):
    def __init__(self):
        super().__init__('test_node')
        self.subscriber = self.create_subscription(
            String, 'test_topic', self.callback, 10)
        self.received_messages = []

    def callback(self, msg):
        self.received_messages.append(msg.data)


class TestRclpy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def test_message_reception(self):
        node = TestNode()
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(node)

        # Publish a test message
        publisher = node.create_publisher(String, 'test_topic', 10)
        test_msg = String()
        test_msg.data = 'test'
        publisher.publish(test_msg)

        # Spin to process messages
        executor.spin_once(timeout_sec=1.0)

        self.assertIn('test', node.received_messages)
        node.destroy_node()
```

## Summary

Advanced rclpy patterns enable you to build robust, scalable, and maintainable ROS 2 applications. Understanding lifecycle nodes, QoS settings, parameter handling, and error handling patterns is crucial for developing production-ready robotic systems.

## Learning Check

After completing this section, you should be able to:
- Implement lifecycle nodes for better resource management
- Configure Quality of Service settings appropriately
- Handle parameters with validation and callbacks
- Create robust error handling in your nodes
- Write effective tests for your ROS 2 components