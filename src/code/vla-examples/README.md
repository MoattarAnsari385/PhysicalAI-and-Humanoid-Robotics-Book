# Vision-Language-Action (VLA) Examples

This directory contains example implementations for the Vision-Language-Action integration module. These examples demonstrate how to integrate multimodal AI models with ROS 2 systems for cognitive robotics applications.

## Directory Structure

```
vla-examples/
├── voice-mapping/        # Voice command processing examples
│   └── whisper_ros_integration.py
├── planning/            # Cognitive planning examples
│   └── cognitive_planner.py
├── capstone/            # Complete capstone scenario
│   └── vla_capstone_demo.py
└── README.md           # This file
```

## Voice Mapping Examples

### whisper_ros_integration.py
Demonstrates integration of OpenAI's Whisper speech recognition model with ROS 2:

- Audio input processing
- Speech-to-text conversion
- Command parsing and mapping
- ROS 2 message publishing

**Usage:**
```bash
python3 whisper_ros_integration.py
```

**Dependencies:**
- `openai-whisper`
- `torch`
- ROS 2 Humble

## Planning Examples

### cognitive_planner.py
Implements cognitive planning for humanoid robots:

- Task decomposition
- Plan generation
- Execution monitoring
- World model management

**Usage:**
```bash
python3 cognitive_planner.py
```

**Dependencies:**
- ROS 2 Humble
- Standard Python libraries

## Capstone Scenario

### vla_capstone_demo.py
Complete integration example demonstrating the full VLA pipeline:

- Voice command processing
- Vision-based perception
- Cognitive planning
- Task execution
- Safety and error handling

**Usage:**
```bash
python3 vla_capstone_demo.py
```

**Dependencies:**
- ROS 2 Humble
- Standard Python libraries

## ROS 2 Message Types

The examples use the following ROS 2 message types:
- `std_msgs/String` - For text commands and status updates
- `sensor_msgs/AudioData` - For audio input (simulated)
- `sensor_msgs/Image` - For visual input
- `geometry_msgs/Twist` - For robot velocity commands
- `geometry_msgs/Pose` - For navigation goals

## Running the Examples

1. Make sure you have ROS 2 Humble installed
2. Install Python dependencies:
   ```bash
   pip install openai-whisper torch
   ```
3. Source your ROS 2 environment:
   ```bash
   source /opt/ros/humble/setup.bash
   ```
4. Run the desired example:
   ```bash
   python3 path/to/example.py
   ```

## Integration with Docusaurus

These examples are referenced in the Module 4 documentation and provide practical implementations of the concepts covered in the VLA integration module.