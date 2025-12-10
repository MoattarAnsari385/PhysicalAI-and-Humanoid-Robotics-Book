---
title: "Module 1 Assessment - ROS 2 Fundamentals"
sidebar_position: 3
description: "Formative assessment for Module 1: Robotic Nervous System (ROS 2)"
---

# Module 1 Assessment - ROS 2 Fundamentals

## Learning Objectives Review

After completing Module 1, you should be able to:
- Understand ROS 2 node architecture and communication patterns
- Implement basic publisher-subscriber communication
- Create and use services for request-response communication
- Work with actions for long-running tasks with feedback
- Build a minimal robot package with proper URDF description

## Assessment Tasks

### Task 1: Publisher-Subscriber Implementation (25 points)

Create a simple publisher node that publishes messages to a custom topic every second. Then create a subscriber node that listens to this topic and logs the received messages.

**Requirements:**
- Publisher should send a message containing a counter value
- Subscriber should log the received message with timestamp
- Both nodes should use proper ROS 2 node structure
- Include proper error handling and cleanup

### Task 2: Service Implementation (25 points)

Implement a service server that calculates the distance between two points in 2D space. Create a client that sends two points to the service and displays the calculated distance.

**Requirements:**
- Define a custom service message with two Point2D requests
- Server should calculate Euclidean distance
- Client should send test points and display result
- Include proper error handling

### Task 3: Robot Package Integration (25 points)

Extend the minimal robot package provided in this module to add a simple sensor. Add a laser scanner to the robot's head and create a node that processes the laser scan data to detect obstacles.

**Requirements:**
- Add a laser scanner sensor to the URDF
- Create a node that subscribes to the laser scan topic
- Implement obstacle detection logic
- Visualize or log obstacle information

### Task 4: System Integration (25 points)

Combine the components from Tasks 1-3 into a single launch file that starts all nodes together. Create a README file explaining how to build and run the complete system.

**Requirements:**
- Single launch file that starts all required nodes
- Proper parameter configuration
- Clear documentation in README
- Working system demonstration

## Grading Rubric

### Functionality (40%)
- All components work as specified
- Proper integration between components
- No critical runtime errors

### Reproducibility (20%)
- Clear build instructions
- All dependencies specified
- Runs in provided development environment

### Code Quality (15%)
- Proper ROS 2 conventions followed
- Clean, well-commented code
- Appropriate error handling

### Documentation & Explanation (15%)
- Clear explanations of design choices
- Proper comments and documentation
- Understanding of concepts demonstrated

### Visuals & Figures (10%)
- Appropriate diagrams showing system architecture
- Clear figures with proper captions
- Good visual representation of concepts

## Submission Requirements

1. Complete source code for all tasks
2. Launch files for system integration
3. Updated URDF with sensor additions
4. README with build and run instructions
5. Brief written explanation of challenges faced and solutions

## Self-Evaluation Questions

Before submitting, answer these questions:

1. Can your publisher-subscriber pair run independently?
2. Does your service handle edge cases properly?
3. Can your robot package be loaded in RViz?
4. Do all nodes properly clean up resources on shutdown?
5. Are your code comments clear and helpful?

## Learning Outcomes Alignment

This assessment addresses the following learning outcomes:
- Understanding of ROS 2 communication patterns
- Practical implementation skills
- System integration capabilities
- Problem-solving in robotics context

## Resources

- ROS 2 Documentation: https://docs.ros.org/en/humble/
- rclpy Tutorials: https://docs.ros.org/en/humble/Tutorials.html
- URDF Tutorials: https://docs.ros.org/en/humble/p/urdf-tutorial.html
- Package Structure: https://docs.ros.org/en/humble/p/ament-python/

## Expected Time Commitment

- Task 1: 2-3 hours
- Task 2: 2-3 hours
- Task 3: 3-4 hours
- Task 4: 1-2 hours
- Total: 8-12 hours

## Support

If you encounter issues:
1. Review the module content and examples
2. Check ROS 2 documentation
3. Use ROS Answers community forum
4. Consult with peers or instructors