#!/usr/bin/env python3
"""
VLA Capstone Scenario: Voice-Controlled Humanoid Robot

This script demonstrates the complete integration of Vision-Language-Action
capabilities in a humanoid robot system. The robot receives voice commands,
processes them using AI models, plans complex behaviors, and executes them.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, AudioData
from geometry_msgs.msg import Twist, Pose
from visualization_msgs.msg import MarkerArray
import json
import time
import threading
import queue
from typing import Dict, Any, Optional


class VLACapstoneNode(Node):
    """
    Main node for the VLA capstone scenario that integrates
    voice processing, vision, planning, and execution.
    """
    def __init__(self):
        super().__init__('vla_capstone_node')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.manipulation_pub = self.create_publisher(String, 'manipulation_commands', 10)
        self.status_pub = self.create_publisher(String, 'capstone_status', 10)
        self.debug_pub = self.create_publisher(MarkerArray, 'capstone_debug', 10)

        # Subscribers
        self.voice_sub = self.create_subscription(
            String, 'vla_commands', self.voice_command_callback, 10)
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10)

        # State management
        self.current_command = None
        self.current_image = None
        self.system_state = 'idle'  # idle, processing, executing, error
        self.world_model = CapstoneWorldModel()
        self.task_planner = CapstoneTaskPlanner(self.world_model)
        self.task_executor = CapstoneTaskExecutor(self, self.world_model)

        # Command queue for processing
        self.command_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self.process_commands, daemon=True)
        self.processing_thread.start()

        self.get_logger().info("VLA Capstone Node initialized")

    def voice_command_callback(self, msg: String):
        """Handle incoming voice commands."""
        try:
            # Parse the structured command
            command_data = json.loads(msg.data)
            self.get_logger().info(f"Received voice command: {command_data}")

            # Add to processing queue
            self.command_queue.put(command_data)

            # Update status
            status_msg = String()
            status_msg.data = f"Received command: {command_data.get('original_text', 'unknown')}"
            self.status_pub.publish(status_msg)

        except json.JSONDecodeError:
            self.get_logger().error("Invalid JSON in voice command")
        except Exception as e:
            self.get_logger().error(f"Error processing voice command: {e}")

    def image_callback(self, msg: Image):
        """Handle incoming image data."""
        # Store current image for processing
        self.current_image = msg
        self.get_logger().debug("Received image data")

    def process_commands(self):
        """Process commands from the queue in a separate thread."""
        while rclpy.ok():
            try:
                # Get command from queue (with timeout)
                command_data = self.command_queue.get(timeout=1.0)

                # Process the command
                self.execute_capstone_scenario(command_data)

                # Mark task as done
                self.command_queue.task_done()

            except queue.Empty:
                # Timeout occurred, continue loop
                continue
            except Exception as e:
                self.get_logger().error(f"Error in command processing thread: {e}")

    def execute_capstone_scenario(self, command_data: Dict[str, Any]):
        """Execute the complete capstone scenario."""
        try:
            self.get_logger().info("Starting capstone scenario execution")

            # Update system state
            self.system_state = 'processing'
            self.publish_status("Processing command")

            # Parse the command to extract task steps
            task_steps = self.task_planner.plan_from_command(command_data)

            self.get_logger().info(f"Generated {len(task_steps)} task steps")

            # Update system state to executing
            self.system_state = 'executing'
            self.publish_status(f"Executing {len(task_steps)} tasks")

            # Execute each step
            for i, step in enumerate(task_steps):
                self.get_logger().info(f"Executing step {i+1}/{len(task_steps)}: {step['action']}")

                success = self.task_executor.execute_task_step(step)

                if not success:
                    self.get_logger().error(f"Task step {i+1} failed: {step}")
                    self.system_state = 'error'
                    self.publish_status(f"Task failed at step {i+1}")
                    return False

                # Update progress
                progress = f"Step {i+1}/{len(task_steps)}: {step['action']}"
                self.publish_status(progress)

            # Scenario completed successfully
            self.system_state = 'idle'
            self.publish_status("Capstone scenario completed successfully")

            self.get_logger().info("Capstone scenario completed successfully")
            return True

        except Exception as e:
            self.get_logger().error(f"Capstone scenario execution error: {e}")
            self.system_state = 'error'
            self.publish_status(f"Error: {str(e)}")
            return False

    def publish_status(self, status: str):
        """Publish status message."""
        status_msg = String()
        status_msg.data = json.dumps({
            'status': status,
            'state': self.system_state,
            'timestamp': time.time()
        })
        self.status_pub.publish(status_msg)


class CapstoneWorldModel:
    """World model for the capstone scenario."""
    def __init__(self):
        self.objects = {}
        self.locations = {
            'kitchen': {'x': 5.0, 'y': 3.0, 'z': 0.0},
            'living_room': {'x': 1.0, 'y': 1.0, 'z': 0.0},
            'office': {'x': 8.0, 'y': 1.0, 'z': 0.0},
            'bedroom': {'x': 3.0, 'y': 5.0, 'z': 0.0}
        }
        self.robot_state = {
            'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'holding': None,
            'battery_level': 1.0
        }
        self.object_colors = {
            'red cup': 'red',
            'blue bottle': 'blue',
            'green book': 'green',
            'white box': 'white'
        }

    def update_object_location(self, obj_id: str, location: Dict[str, float]):
        """Update the location of an object."""
        if obj_id not in self.objects:
            self.objects[obj_id] = {}
        self.objects[obj_id]['location'] = location
        self.objects[obj_id]['detected'] = True

    def get_object_location(self, obj_id: str) -> Optional[Dict[str, float]]:
        """Get the location of an object."""
        if obj_id in self.objects:
            return self.objects[obj_id].get('location')
        return None

    def get_objects_at_location(self, location_name: str) -> list:
        """Get all objects at a specific location."""
        location = self.locations.get(location_name)
        if not location:
            return []

        objects_here = []
        for obj_id, obj_data in self.objects.items():
            if 'location' in obj_data:
                obj_loc = obj_data['location']
                # Simple distance check (in a real system, this would be more sophisticated)
                distance = ((location['x'] - obj_loc['x'])**2 +
                           (location['y'] - obj_loc['y'])**2)**0.5
                if distance < 1.0:  # Within 1 meter
                    objects_here.append(obj_id)

        return objects_here

    def update_robot_position(self, position: Dict[str, float]):
        """Update the robot's position."""
        self.robot_state['position'] = position

    def update_robot_holding(self, obj_id: Optional[str]):
        """Update what the robot is holding."""
        self.robot_state['holding'] = obj_id


class CapstoneTaskPlanner:
    """Task planner for the capstone scenario."""
    def __init__(self, world_model: CapstoneWorldModel):
        self.world_model = world_model

    def plan_from_command(self, command_data: Dict[str, Any]) -> list:
        """Generate task plan from voice command data."""
        entities = command_data.get('entities', {})
        cmd_type = command_data.get('type', 'unknown')

        steps = []

        # Based on command type and entities, create task steps
        if cmd_type == 'delivery' and len(entities.get('locations', [])) >= 2:
            # Delivery command: go to location 1, get object, go to location 2, deliver
            start_location = entities['locations'][0]
            end_location = entities['locations'][1]

            if entities.get('objects'):
                target_object = entities['objects'][0]

                steps.extend([
                    {
                        'action': 'navigate',
                        'target_location': start_location,
                        'description': f'Navigate to {start_location}'
                    },
                    {
                        'action': 'detect_object',
                        'target_object': target_object,
                        'description': f'Detect {target_object} in {start_location}'
                    },
                    {
                        'action': 'grasp_object',
                        'target_object': target_object,
                        'description': f'Grasp {target_object}'
                    },
                    {
                        'action': 'navigate',
                        'target_location': end_location,
                        'description': f'Navigate to {end_location} with {target_object}'
                    },
                    {
                        'action': 'release_object',
                        'target_object': target_object,
                        'description': f'Release {target_object} at {end_location}'
                    }
                ])
        elif cmd_type == 'fetch_object' and entities.get('locations'):
            # Fetch command: go to location, get object, return
            location = entities['locations'][0]

            if entities.get('objects'):
                target_object = entities['objects'][0]

                steps.extend([
                    {
                        'action': 'navigate',
                        'target_location': location,
                        'description': f'Navigate to {location}'
                    },
                    {
                        'action': 'detect_object',
                        'target_object': target_object,
                        'description': f'Detect {target_object} in {location}'
                    },
                    {
                        'action': 'grasp_object',
                        'target_object': target_object,
                        'description': f'Grasp {target_object}'
                    },
                    {
                        'action': 'navigate',
                        'target_location': 'return_point',  # Could be current position or specified
                        'description': f'Return with {target_object}'
                    }
                ])
        elif cmd_type == 'navigation' and entities.get('locations'):
            # Simple navigation command
            location = entities['locations'][0]

            steps.append({
                'action': 'navigate',
                'target_location': location,
                'description': f'Navigate to {location}'
            })
        else:
            # Default: try to understand and create appropriate steps
            steps = self.create_default_plan(command_data)

        return steps

    def create_default_plan(self, command_data: Dict[str, Any]) -> list:
        """Create a default plan when command type is unclear."""
        entities = command_data.get('entities', {})
        original_text = command_data.get('original_text', '').lower()

        steps = []

        # Look for keywords to determine intent
        if 'kitchen' in original_text or 'living room' in original_text:
            # Likely navigation command
            for location in ['kitchen', 'living room', 'office', 'bedroom']:
                if location in original_text:
                    steps.append({
                        'action': 'navigate',
                        'target_location': location,
                        'description': f'Navigate to {location}'
                    })
                    break

        if 'cup' in original_text or 'bottle' in original_text or 'book' in original_text:
            # Likely manipulation command
            for obj in ['cup', 'bottle', 'book']:
                if obj in original_text:
                    steps.append({
                        'action': 'detect_object',
                        'target_object': obj,
                        'description': f'Detect {obj}'
                    })
                    steps.append({
                        'action': 'grasp_object',
                        'target_object': obj,
                        'description': f'Grasp {obj}'
                    })
                    break

        return steps


class CapstoneTaskExecutor:
    """Task executor for the capstone scenario."""
    def __init__(self, node: Node, world_model: CapstoneWorldModel):
        self.node = node
        self.world_model = world_model

    def execute_task_step(self, step: Dict[str, Any]) -> bool:
        """Execute a single task step."""
        action = step['action']

        try:
            if action == 'navigate':
                return self.execute_navigation_step(step)
            elif action == 'detect_object':
                return self.execute_detection_step(step)
            elif action == 'grasp_object':
                return self.execute_grasp_step(step)
            elif action == 'release_object':
                return self.execute_release_step(step)
            else:
                self.node.get_logger().error(f"Unknown action: {action}")
                return False
        except Exception as e:
            self.node.get_logger().error(f"Error executing task step: {e}")
            return False

    def execute_navigation_step(self, step: Dict[str, Any]) -> bool:
        """Execute navigation step."""
        target_location = step['target_location']

        if target_location in self.world_model.locations:
            target_pose = self.world_model.locations[target_location]

            self.node.get_logger().info(f"Navigating to {target_location} at {target_pose}")

            # Simulate navigation
            cmd = Twist()
            cmd.linear.x = 0.5  # Move forward
            cmd.angular.z = 0.0

            # Publish command for 3 seconds (simulated navigation)
            for _ in range(30):  # 3 seconds at 10Hz
                self.node.cmd_vel_pub.publish(cmd)
                time.sleep(0.1)

            # Stop robot
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.node.cmd_vel_pub.publish(cmd)

            # Update world model
            self.world_model.update_robot_position(target_pose)

            self.node.get_logger().info(f"Reached {target_location}")
            return True
        else:
            self.node.get_logger().error(f"Unknown location: {target_location}")
            return False

    def execute_detection_step(self, step: Dict[str, Any]) -> bool:
        """Execute object detection step."""
        target_object = step['target_object']

        self.node.get_logger().info(f"Detecting {target_object}")

        # Simulate detection process
        time.sleep(1.5)

        # Check if object exists in world model, if not, add it at current location
        if target_object not in self.world_model.objects:
            current_pos = self.world_model.robot_state['position']
            self.world_model.update_object_location(target_object, current_pos)
            self.node.get_logger().info(f"Added {target_object} to world model at current position")

        self.node.get_logger().info(f"Detected {target_object}")
        return True

    def execute_grasp_step(self, step: Dict[str, Any]) -> bool:
        """Execute object grasping step."""
        target_object = step['target_object']

        self.node.get_logger().info(f"Grasping {target_object}")

        # Simulate grasping
        time.sleep(1.0)

        # Update world model
        self.world_model.update_robot_holding(target_object)

        # Publish manipulation command
        manip_msg = String()
        manip_msg.data = f"grasp:{target_object}"
        self.node.manipulation_pub.publish(manip_msg)

        self.node.get_logger().info(f"Successfully grasped {target_object}")
        return True

    def execute_release_step(self, step: Dict[str, Any]) -> bool:
        """Execute object release step."""
        target_object = step['target_object']

        self.node.get_logger().info(f"Releasing {target_object}")

        # Simulate releasing
        time.sleep(1.0)

        # Update world model
        self.world_model.update_robot_holding(None)

        # Publish manipulation command
        manip_msg = String()
        manip_msg.data = f"release:{target_object}"
        self.node.manipulation_pub.publish(manip_msg)

        self.node.get_logger().info(f"Successfully released {target_object}")
        return True


def main(args=None):
    """Main function to run the VLA capstone node."""
    rclpy.init(args=args)

    vla_capstone = VLACapstoneNode()

    # Example: Simulate a voice command after startup
    def simulate_voice_command():
        """Simulate a voice command after the node starts."""
        time.sleep(2)  # Wait for node to initialize

        # Example command: "Go to the kitchen, find the red cup, pick it up, and bring it to the living room"
        command_data = {
            'type': 'delivery',
            'original_text': 'Go to the kitchen, find the red cup, pick it up, and bring it to the living room',
            'entities': {
                'locations': ['kitchen', 'living room'],
                'objects': ['red cup'],
                'people': []
            },
            'timestamp': time.time()
        }

        command_msg = String()
        command_msg.data = json.dumps(command_data)

        vla_capstone.get_logger().info("Simulating voice command")
        vla_capstone.voice_command_callback(command_msg)

    # Start simulation in a separate thread
    sim_thread = threading.Thread(target=simulate_voice_command, daemon=True)
    sim_thread.start()

    try:
        rclpy.spin(vla_capstone)
    except KeyboardInterrupt:
        pass
    finally:
        vla_capstone.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()