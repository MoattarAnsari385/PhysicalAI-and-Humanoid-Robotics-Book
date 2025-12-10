#!/usr/bin/env python3
"""
Whisper Voice-to-ROS Command Mapping Example

This script demonstrates how to integrate OpenAI's Whisper speech recognition
model with ROS 2 to convert voice commands into robotic actions.

The system includes:
- Audio input processing
- Speech-to-text conversion using Whisper
- Command parsing and mapping
- ROS 2 message publishing
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import AudioData
from geometry_msgs.msg import Twist
import whisper
import torch
import numpy as np
import threading
import queue
import re


class WhisperNode(Node):
    """
    ROS 2 node that processes audio input using Whisper and maps
    voice commands to ROS 2 messages.
    """
    def __init__(self):
        super().__init__('whisper_node')

        # Create subscribers
        self.subscription = self.create_subscription(
            AudioData,
            'audio_input',
            self.audio_callback,
            10)

        # Create publishers
        self.command_publisher = self.create_publisher(String, 'voice_commands', 10)
        self.feedback_publisher = self.create_publisher(String, 'voice_feedback', 10)
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        # Load Whisper model (using 'tiny' model for efficiency)
        self.get_logger().info("Loading Whisper model...")
        try:
            self.model = whisper.load_model("tiny")
            self.get_logger().info("Whisper model loaded successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to load Whisper model: {e}")
            self.model = None

        # Command mapping dictionary
        self.command_map = {
            'move forward': self.move_forward,
            'move backward': self.move_backward,
            'turn left': self.turn_left,
            'turn right': self.turn_right,
            'stop': self.stop_robot,
            'go to kitchen': lambda: self.navigate_to('kitchen'),
            'go to living room': lambda: self.navigate_to('living_room'),
        }

        # Audio processing queue
        self.audio_queue = queue.Queue(maxsize=10)
        self.processing_thread = threading.Thread(target=self.process_audio_queue, daemon=True)
        self.processing_thread.start()

    def audio_callback(self, msg):
        """Callback function for audio input."""
        try:
            # Add audio data to processing queue
            if not self.audio_queue.full():
                self.audio_queue.put(msg)
            else:
                self.get_logger().warn("Audio queue is full, dropping audio packet")
        except Exception as e:
            self.get_logger().error(f"Error in audio callback: {e}")

    def process_audio_queue(self):
        """Process audio data from the queue in a separate thread."""
        while rclpy.ok():
            try:
                # Get audio data from queue (with timeout to allow graceful shutdown)
                audio_msg = self.audio_queue.get(timeout=1.0)

                if self.model is not None:
                    # Convert audio data to numpy array
                    audio_array = self.convert_audio_data(audio_msg)

                    # Transcribe audio to text
                    result = self.model.transcribe(audio_array)
                    transcription = result['text'].strip()

                    if transcription:
                        self.get_logger().info(f"Transcribed: {transcription}")

                        # Publish transcription as command
                        cmd_msg = String()
                        cmd_msg.data = transcription
                        self.command_publisher.publish(cmd_msg)

                        # Process command
                        self.process_command(transcription)

                        # Provide feedback
                        feedback_msg = String()
                        feedback_msg.data = f"Heard: {transcription}"
                        self.feedback_publisher.publish(feedback_msg)

                self.audio_queue.task_done()

            except queue.Empty:
                # Timeout occurred, continue loop
                continue
            except Exception as e:
                self.get_logger().error(f"Error processing audio: {e}")

    def convert_audio_data(self, audio_msg):
        """Convert ROS AudioData message to numpy array."""
        # Convert byte data to numpy array
        # Assuming 16-bit signed integers (common format)
        audio_array = np.frombuffer(audio_msg.data, dtype=np.int16)

        # Normalize to float32 in range [-1, 1]
        audio_array = audio_array.astype(np.float32) / 32768.0

        return audio_array

    def process_command(self, command_text):
        """Process the transcribed command and execute appropriate action."""
        command_text_lower = command_text.lower()

        # Find matching command
        for cmd_key, cmd_func in self.command_map.items():
            if cmd_key in command_text_lower:
                self.get_logger().info(f"Executing command: {cmd_key}")
                cmd_func()
                return

        # If no command matched, log the unrecognized text
        self.get_logger().info(f"Unrecognized command: {command_text}")

    def move_forward(self):
        """Move robot forward."""
        cmd = Twist()
        cmd.linear.x = 0.5  # Forward velocity
        cmd.angular.z = 0.0
        self.cmd_vel_publisher.publish(cmd)

    def move_backward(self):
        """Move robot backward."""
        cmd = Twist()
        cmd.linear.x = -0.5  # Backward velocity
        cmd.angular.z = 0.0
        self.cmd_vel_publisher.publish(cmd)

    def turn_left(self):
        """Turn robot left."""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.5  # Left turn
        self.cmd_vel_publisher.publish(cmd)

    def turn_right(self):
        """Turn robot right."""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = -0.5  # Right turn
        self.cmd_vel_publisher.publish(cmd)

    def stop_robot(self):
        """Stop robot movement."""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_publisher.publish(cmd)

    def navigate_to(self, location):
        """Navigate to specified location (placeholder implementation)."""
        self.get_logger().info(f"Planning navigation to {location}")
        # In a real implementation, this would call navigation services
        # For now, just log the intent
        pass


def main(args=None):
    """Main function to run the Whisper node."""
    rclpy.init(args=args)

    whisper_node = WhisperNode()

    try:
        rclpy.spin(whisper_node)
    except KeyboardInterrupt:
        pass
    finally:
        whisper_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()