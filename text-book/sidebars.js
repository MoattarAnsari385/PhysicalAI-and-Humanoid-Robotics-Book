// @ts-check

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.

 @type {import('@docusaurus/plugin-content-docs').SidebarsConfig}
 */
const sidebars = {
  // Manual sidebar for the Physical AI & Humanoid Robotics Textbook
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Introduction',
      items: ['intro'],
    },
    {
      type: 'category',
      label: 'Module 1: Robotic Nervous System (ROS 2)',
      items: [
        'module-1-ros2/intro',
        'module-1-ros2/nodes-topics-services',
        'module-1-ros2/ros2-architecture',
        'module-1-ros2/urdf-basics',
        'module-1-ros2/rclpy-examples',
        'module-1-ros2/minimal-robot-package',
        'module-1-ros2/assessments/module1-assessment',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Simulation)',
      items: [
        'module-2-simulation/intro',
        'module-2-simulation/gazebo-worlds',
        'module-2-simulation/physics-simulation',
        'module-2-simulation/sensors',
        'module-2-simulation/humanoid-test-scene',
        'module-2-simulation/assessments/module2-assessment',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac)',
      items: [
        'module-3-ai/intro',
        'module-3-ai/isaac-overview',
        'module-3-ai/perception-pipelines',
        'module-3-ai/navigation-workflows',
        'module-3-ai/object-detection',
        'module-3-ai/isaac-sim',
        'module-3-ai/synthetic-datasets',
        'module-3-ai/assessments/module3-assessments',
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      items: [
        'module-4-vla/intro',
        'module-4-vla/voice-mapping',
        'module-4-vla/cognitive-planning',
        'module-4-vla/vla-integration',
        'module-4-vla/task-planning',
        'module-4-vla/capstone-scenario',
        'module-4-vla/assessments/module4-assessments',
      ],
    },
  ],
};

export default sidebars;
