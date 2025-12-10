import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import math


class RobotController(Node):

    def __init__(self):
        super().__init__('robot_controller')

        # Declare parameters with defaults
        self.declare_parameter('linear_velocity', 0.5)
        self.declare_parameter('angular_velocity', 0.5)
        self.declare_parameter('obstacle_detection_distance', 1.0)
        self.declare_parameter('robot_name', 'minimal_robot')

        # Get parameter values
        self.linear_velocity = self.get_parameter('linear_velocity').value
        self.angular_velocity = self.get_parameter('angular_velocity').value
        self.obstacle_distance = self.get_parameter('obstacle_detection_distance').value
        self.robot_name = self.get_parameter('robot_name').value

        # Publisher for robot velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscriber for laser scan data
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)

        # Subscriber for odometry data
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)

        # Timer for control loop
        self.timer = self.create_timer(0.1, self.control_loop)

        # Robot state
        self.current_pose = None
        self.laser_data = None
        self.obstacle_detected = False

        self.get_logger().info(f'Robot Controller initialized for {self.robot_name}')
        self.get_logger().info(f'Linear velocity: {self.linear_velocity}, Angular velocity: {self.angular_velocity}')

    def scan_callback(self, msg):
        """Process laser scan data to detect obstacles"""
        self.laser_data = msg
        # Check for obstacles in front of robot
        if msg.ranges:
            front_ranges = msg.ranges[:30] + msg.ranges[-30:]  # Front 60 degrees
            min_distance = min([r for r in front_ranges if r != float('inf') and r > 0], default=float('inf'))
            self.obstacle_detected = min_distance < self.obstacle_distance
            if self.obstacle_detected:
                self.get_logger().info(f'Obstacle detected at {min_distance:.2f}m')

    def odom_callback(self, msg):
        """Process odometry data"""
        self.current_pose = msg.pose.pose

    def control_loop(self):
        """Main control loop"""
        if self.obstacle_detected:
            # Turn if obstacle detected
            self.move_robot(0.0, self.angular_velocity)
        else:
            # Move forward if no obstacles
            self.move_robot(self.linear_velocity, 0.0)

    def move_robot(self, linear_vel, angular_vel):
        """Send velocity commands to robot"""
        msg = Twist()
        msg.linear.x = linear_vel
        msg.angular.z = angular_vel
        self.cmd_vel_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    robot_controller = RobotController()

    try:
        rclpy.spin(robot_controller)
    except KeyboardInterrupt:
        robot_controller.get_logger().info('Shutting down Robot Controller...')
    finally:
        robot_controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()