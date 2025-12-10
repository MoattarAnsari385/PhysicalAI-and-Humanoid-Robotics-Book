# Minimal Robot Package

This package serves as a minimal example for the Physical AI & Humanoid Robotics textbook. It demonstrates fundamental ROS 2 concepts including nodes, topics, services, and URDF.

## Overview

The package includes:
- Basic publisher and subscriber nodes
- Robot controller node
- Humanoid skeleton URDF
- Launch file to start the complete system

## Installation

1. Clone this repository into your ROS 2 workspace `src` directory
2. Navigate to your workspace root
3. Build the package:

```bash
colcon build --packages-select minimal_robot_package
source install/setup.bash
```

## Usage

### Launch the complete system:

```bash
ros2 launch minimal_robot_package minimal_robot.launch.py
```

### Run individual nodes:

```bash
# Publisher node
ros2 run minimal_robot_package minimal_publisher

# Subscriber node
ros2 run minimal_robot_package minimal_subscriber

# Robot controller
ros2 run minimal_robot_package robot_controller
```

## Package Structure

```
minimal_robot_package/
├── package.xml          # Package metadata
├── setup.py            # Python setup configuration
├── minimal_robot_package/ # Python modules
│   ├── __init__.py
│   ├── minimal_publisher.py
│   ├── minimal_subscriber.py
│   └── robot_controller.py
├── launch/             # Launch files
│   └── minimal_robot.launch.py
├── urdf/               # Robot description
│   └── humanoid_skeleton.urdf
└── README.md
```

## Nodes

### `minimal_publisher`
- Publishes "Hello World" messages to the `/topic` topic
- Demonstrates basic publisher implementation

### `minimal_subscriber`
- Subscribes to the `/topic` topic
- Logs received messages to console

### `robot_controller`
- Subscribes to laser scan data (`/scan`)
- Subscribes to odometry data (`/odom`)
- Publishes velocity commands (`/cmd_vel`)
- Implements simple obstacle avoidance behavior

## Topics

- `/topic`: Simple string messages for publisher/subscriber example
- `/cmd_vel`: Velocity commands for robot movement
- `/scan`: Laser scan data (requires simulation environment)
- `/odom`: Odometry data (requires simulation environment)

## URDF Model

The humanoid skeleton URDF defines a simple bipedal robot with:
- Base link
- Torso
- Head
- Two arms (upper and lower)
- Two legs (upper and lower)
- Proper joint limits and physical properties

## Launch File

The launch file starts:
- Robot State Publisher to publish the URDF
- All three example nodes
- Proper parameter configuration

## Learning Objectives

This package demonstrates:
1. ROS 2 node structure and lifecycle
2. Publisher-subscriber communication pattern
3. Basic robot control concepts
4. URDF robot description format
5. Launch file coordination

## Troubleshooting

- Ensure the workspace is properly sourced (`source install/setup.bash`)
- Check that required dependencies are installed
- Verify topic names match between publishers and subscribers

## License

This package is released under the Apache 2.0 license.