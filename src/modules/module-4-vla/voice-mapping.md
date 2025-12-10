---
sidebar_position: 2
title: "Whisper Voice-to-ROS Command Mapping"
---

# Whisper Voice-to-ROS Command Mapping

## Overview

Voice command mapping is a critical component of cognitive robotics that enables natural human-robot interaction. This section explores the integration of OpenAI's Whisper speech recognition model with ROS 2 systems to convert spoken commands into robotic actions. The implementation involves real-time speech processing, natural language understanding, and command execution within the ROS 2 ecosystem.

## Learning Objectives

By the end of this section, you will be able to:
- Set up Whisper speech recognition within a ROS 2 environment
- Design voice command grammars for robotic control
- Map natural language commands to ROS 2 topics and services
- Implement error handling and feedback mechanisms for voice commands
- Integrate voice processing with robot behavior trees

## Whisper Speech Recognition Integration

Whisper is a state-of-the-art speech recognition model developed by OpenAI that provides high accuracy across multiple languages and audio conditions. For robotic applications, Whisper can be integrated as a ROS 2 node that processes audio input and generates text commands for the robot's cognitive system.

### Audio Input Pipeline

The audio input pipeline connects microphones or audio streams to the Whisper processing node. This involves:

1. **Audio Capture**: Real-time audio sampling from microphone arrays or audio streams
2. **Preprocessing**: Noise reduction, audio normalization, and format conversion
3. **Transcription**: Speech-to-text conversion using Whisper models
4. **Command Parsing**: Natural language processing to extract actionable commands

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import AudioData
import whisper
import torch

class WhisperNode(Node):
    def __init__(self):
        super().__init__('whisper_node')
        self.subscription = self.create_subscription(
            AudioData,
            'audio_input',
            self.audio_callback,
            10)
        self.command_publisher = self.create_publisher(String, 'voice_commands', 10)

        # Load Whisper model
        self.model = whisper.load_model("base")

    def audio_callback(self, msg):
        # Convert audio data to format expected by Whisper
        audio_array = self.process_audio_data(msg.data)

        # Transcribe audio to text
        result = self.model.transcribe(audio_array)
        transcription = result['text']

        # Publish transcription as command
        cmd_msg = String()
        cmd_msg.data = transcription
        self.command_publisher.publish(cmd_msg)
```

### Command Grammar Design

Effective voice command mapping requires well-designed grammars that balance natural language flexibility with reliable command recognition. Key considerations include:

- **Command Structure**: Consistent patterns for voice commands (e.g., "Robot, move to the kitchen")
- **Synonym Handling**: Multiple ways to express the same command
- **Context Awareness**: Commands that depend on robot state or environment
- **Error Recovery**: Mechanisms for handling unrecognized or ambiguous commands

## ROS 2 Integration Patterns

### Topic-Based Command System

The most straightforward approach maps voice commands to ROS 2 topics for simple actions:

```yaml
# Voice command mapping configuration
command_mappings:
  "move forward":
    topic: "/cmd_vel"
    message_type: "geometry_msgs/Twist"
    values: {"linear.x": 0.5, "angular.z": 0.0}
  "turn left":
    topic: "/cmd_vel"
    message_type: "geometry_msgs/Twist"
    values: {"linear.x": 0.0, "angular.z": 0.5}
```

### Service-Based Command System

For more complex commands that require responses, service calls provide synchronous execution:

```python
class VoiceCommandService(Node):
    def __init__(self):
        super().__init__('voice_command_service')
        self.srv = self.create_service(String, 'execute_voice_command', self.execute_command)

    def execute_command(self, request, response):
        command = request.data
        # Parse and execute command
        result = self.parse_and_execute(command)
        response.data = result
        return response
```

## Natural Language Processing

### Intent Recognition

Intent recognition identifies the user's goal from spoken commands. Common intents for robotic systems include:

- **Navigation**: Commands to move the robot to specific locations
- **Manipulation**: Commands to pick up, move, or interact with objects
- **Information**: Requests for robot status or environmental data
- **Control**: Commands to start, stop, or modify robot behavior

### Entity Extraction

Entity extraction identifies specific objects, locations, or parameters mentioned in voice commands:

```python
import spacy

class EntityExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def extract_entities(self, text):
        doc = self.nlp(text)
        entities = {}

        for ent in doc.ents:
            if ent.label_ in ["LOCATION", "OBJECT", "PERSON"]:
                entities[ent.label_.lower()] = ent.text

        return entities
```

## Error Handling and Feedback

### Confidence Scoring

Whisper provides confidence scores that can be used to determine if a transcription should be processed:

```python
def process_transcription(self, result):
    if result['confidence'] > 0.8:  # Confidence threshold
        return self.parse_command(result['text'])
    else:
        self.publish_feedback("Command not understood, please repeat")
        return None
```

### Voice Feedback System

A voice feedback system provides confirmation and error messages to users:

```python
class VoiceFeedbackNode(Node):
    def __init__(self):
        super().__init__('voice_feedback')
        self.feedback_publisher = self.create_publisher(String, 'tts_input', 10)

    def confirm_command(self, command):
        feedback_msg = String()
        feedback_msg.data = f"Executing command: {command}"
        self.feedback_publisher.publish(feedback_msg)

    def error_feedback(self, error_msg):
        feedback_msg = String()
        feedback_msg.data = f"Error: {error_msg}"
        self.feedback_publisher.publish(feedback_msg)
```

## Practical Implementation Example

Here's a complete example of a voice command system for a humanoid robot:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from builtin_interfaces.msg import Duration
import whisper
import threading
import queue

class HumanoidVoiceController(Node):
    def __init__(self):
        super().__init__('humanoid_voice_controller')

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.voice_cmd_pub = self.create_publisher(String, '/voice_commands', 10)
        self.feedback_pub = self.create_publisher(String, '/voice_feedback', 10)

        # Voice command subscription
        self.voice_sub = self.create_subscription(
            String, '/voice_commands', self.voice_command_callback, 10)

        # Whisper model
        self.whisper_model = whisper.load_model("base")

        # Command mapping dictionary
        self.command_map = {
            'move forward': self.move_forward,
            'move backward': self.move_backward,
            'turn left': self.turn_left,
            'turn right': self.turn_right,
            'stop': self.stop_robot,
            'walk to kitchen': lambda: self.navigate_to('kitchen'),
            'walk to living room': lambda: self.navigate_to('living_room'),
        }

        self.get_logger().info("Humanoid Voice Controller initialized")

    def voice_command_callback(self, msg):
        command = msg.data.lower().strip()
        self.get_logger().info(f"Received voice command: {command}")

        # Find matching command
        for cmd_key, cmd_func in self.command_map.items():
            if cmd_key in command:
                cmd_func()
                self.publish_feedback(f"Executing: {cmd_key}")
                return

        # Command not recognized
        self.publish_feedback(f"Command not recognized: {command}")

    def move_forward(self):
        cmd = Twist()
        cmd.linear.x = 0.5
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)

    def move_backward(self):
        cmd = Twist()
        cmd.linear.x = -0.5
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)

    def turn_left(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.5
        self.cmd_vel_pub.publish(cmd)

    def turn_right(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = -0.5
        self.cmd_vel_pub.publish(cmd)

    def stop_robot(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)

    def navigate_to(self, location):
        # This would typically call a navigation service
        self.get_logger().info(f"Planning navigation to {location}")
        # Implementation would depend on navigation stack
        pass

    def publish_feedback(self, message):
        feedback_msg = String()
        feedback_msg.data = message
        self.feedback_pub.publish(feedback_msg)

def main(args=None):
    rclpy.init(args=args)
    voice_controller = HumanoidVoiceController()

    try:
        rclpy.spin(voice_controller)
    except KeyboardInterrupt:
        pass
    finally:
        voice_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Performance Considerations

### Real-time Processing

Voice processing systems must operate in real-time to provide responsive interaction. Key performance considerations include:

- **Latency**: Minimize delay between speech input and command execution
- **Throughput**: Handle multiple commands efficiently
- **Resource Usage**: Optimize model inference for embedded systems

### Model Optimization

For deployment on robotic platforms, Whisper models can be optimized:

- **Model Quantization**: Reduce model size while maintaining accuracy
- **Onnx Conversion**: Convert to optimized inference formats
- **Hardware Acceleration**: Utilize GPU or specialized AI chips

## Security and Privacy

Voice processing systems must consider privacy and security:

- **Data Encryption**: Encrypt audio data in transit and at rest
- **Access Control**: Limit access to voice processing capabilities
- **Privacy Preservation**: Implement local processing where possible

## Summary

Voice-to-ROS command mapping enables natural human-robot interaction through speech recognition and natural language processing. The integration of Whisper with ROS 2 systems provides a robust foundation for voice-controlled robotic applications. Proper design of command grammars, error handling, and feedback systems ensures reliable and intuitive operation.

The next section will explore cognitive planning systems that interpret voice commands and generate appropriate robotic behaviors.