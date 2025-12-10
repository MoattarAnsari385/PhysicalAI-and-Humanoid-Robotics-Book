---
title: "Physics Simulation and Dynamics"
sidebar_position: 2
description: "Understanding physics simulation in Gazebo including dynamics, collision detection, and material properties"
---

# Physics Simulation and Dynamics

## Introduction to Physics Simulation

Physics simulation is the cornerstone of realistic robot simulation. In Gazebo, the physics engine handles collision detection, rigid body dynamics, and contact physics. Understanding these concepts is crucial for creating realistic simulations that accurately reflect real-world behavior.

## Physics Engines in Gazebo

Gazebo supports multiple physics engines, each with different characteristics:

### Open Dynamics Engine (ODE)
- **Pros**: Fast, stable, widely used
- **Cons**: Less accurate for complex contact scenarios
- **Best for**: General-purpose simulation, real-time applications

### Bullet Physics
- **Pros**: More accurate contact simulation, better for complex interactions
- **Cons**: Slower than ODE
- **Best for**: Applications requiring high-fidelity contact physics

### Simbody
- **Pros**: Very accurate for complex articulated systems
- **Cons**: Complex to configure, slower performance
- **Best for**: Multi-body dynamics with complex joints

## Physics Configuration Parameters

### Time Step Settings
The time step determines the granularity of the simulation:

```xml
<physics type="ode">
  <max_step_size>0.001</max_step_size>  <!-- Physics update interval (seconds) -->
  <real_time_update_rate>1000</real_time_update_rate>  <!-- Updates per second -->
  <real_time_factor>1</real_time_factor>  <!-- Simulation speed multiplier -->
</physics>
```

**Key considerations:**
- Smaller step sizes = more accurate but slower simulation
- Real-time factor of 1.0 = simulation runs at real-world speed
- Higher real-time update rates = smoother simulation but more CPU intensive

### Solver Parameters
The physics solver handles the mathematical calculations:

```xml
<physics type="ode">
  <!-- ODE-specific parameters -->
  <ode>
    <solver>
      <type>quick</type>  <!-- Type of solver: world, quick -->
      <iters>10</iters>   <!-- Number of iterations per step -->
      <sor>1.3</sor>      <!-- Successive over-relaxation parameter -->
    </solver>
    <constraints>
      <cfm>0.000001</cfm>  <!-- Constraint force mixing parameter -->
      <erp>0.2</erp>       <!-- Error reduction parameter -->
      <contact_max_correcting_vel>100</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

## Rigid Body Dynamics

### Mass and Inertia
Proper mass and inertia properties are critical for realistic simulation:

```xml
<inertial>
  <mass>1.0</mass>
  <inertia>
    <ixx>0.166667</ixx>
    <ixy>0</ixy>
    <ixz>0</ixz>
    <iyy>0.166667</iyy>
    <iyz>0</iyz>
    <izz>0.166667</izz>
  </inertia>
</inertial>
```

For a uniform box with mass m and dimensions (x, y, z):
- Ixx = m * (y² + z²) / 12
- Iyy = m * (x² + z²) / 12
- Izz = m * (x² + y²) / 12

### Center of Mass
The center of mass should be correctly positioned relative to the link:

```xml
<inertial>
  <mass>1.0</mass>
  <pose>0.1 0 0 0 0 0</pose>  <!-- Offset from link origin -->
  <inertia>
    <!-- ... -->
  </inertia>
</inertial>
```

## Collision Detection

### Collision Geometry Types
Different collision geometries affect performance and accuracy:

```xml
<collision name="collision">
  <!-- Simple geometries (fast) -->
  <geometry>
    <box>
      <size>1 1 1</size>
    </box>
  </geometry>
</collision>

<collision name="collision">
  <!-- Complex mesh (slow but accurate) -->
  <geometry>
    <mesh>
      <uri>file://meshes/complex_shape.dae</uri>
    </mesh>
  </geometry>
</collision>
```

**Geometry types:**
- **Box**: Fastest, good for simple shapes
- **Cylinder**: Good for wheels, limbs
- **Sphere**: Fast, good for balls, rounded objects
- **Mesh**: Most accurate but slowest
- **Plane**: For infinite ground planes

### Contact Parameters
Fine-tune contact behavior:

```xml
<collision name="collision">
  <surface>
    <contact>
      <ode>
        <max_vel>100</max_vel>          <!-- Maximum contact penetration velocity -->
        <min_depth>0.001</min_depth>    <!-- Minimum contact depth -->
      </ode>
    </contact>
    <friction>
      <ode>
        <mu>1.0</mu>                   <!-- Static friction coefficient -->
        <mu2>1.0</mu2>                 <!-- Secondary friction coefficient -->
        <fdir1>0 0 1</fdir1>           <!-- Friction direction -->
      </ode>
    </friction>
    <bounce>
      <restitution_coefficient>0.1</restitution_coefficient>  <!-- Bounciness -->
      <threshold>100000</threshold>                            <!-- Velocity threshold -->
    </bounce>
  </surface>
</collision>
```

## Material Properties and Surface Interactions

### Friction Modeling
Friction is critical for realistic locomotion:

```xml
<!-- High friction surface (good for walking) -->
<surface>
  <friction>
    <ode>
      <mu>1.0</mu>    <!-- Static friction -->
      <mu2>1.0</mu2>  <!-- Dynamic friction -->
    </ode>
  </friction>
</surface>

<!-- Low friction surface (challenging for walking) -->
<surface>
  <friction>
    <ode>
      <mu>0.1</mu>
      <mu2>0.1</mu2>
    </ode>
  </friction>
</surface>
```

### Damping
Damping helps stabilize simulations and model energy loss:

```xml
<inertial>
  <mass>1.0</mass>
  <inertia>
    <!-- ... -->
  </inertia>
  <!-- Linear and angular damping to simulate air resistance, etc. -->
  <linear_damping>0.01</linear_damping>
  <angular_damping>0.01</angular_damping>
</inertial>
```

## Humanoid-Specific Physics Considerations

### Balancing and Stability
Humanoid robots require special attention to physics properties:

```xml
<!-- Lower body links should have appropriate mass distribution -->
<link name="left_upper_leg">
  <inertial>
    <mass>0.8</mass>  <!-- Heavier than arms for stability -->
    <pose>0 0 -0.2 0 0 0</pose>  <!-- COM toward center of limb -->
    <inertia>
      <ixx>0.02</ixx>
      <iyy>0.02</iyy>
      <izz>0.005</izz>
      <!-- Off-diagonal terms for realistic mass distribution -->
      <ixy>0</ixy>
      <ixz>0.005</ixz>
      <iyz>0</iyz>
    </inertia>
  </inertial>
</link>
```

### Joint Constraints and Limits
Proper joint limits prevent unrealistic movements:

```xml
<joint name="left_knee" type="revolute">
  <parent link="left_upper_leg"/>
  <child link="left_lower_leg"/>
  <origin xyz="0 0 -0.4" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>  <!-- Rotate about Y axis -->
  <limit lower="0" upper="2.5" effort="100" velocity="2"/>
  <!-- Spring and damping for more realistic joint behavior -->
  <dynamics damping="0.1" friction="0.01"/>
</joint>
```

## Advanced Physics Concepts

### Multi-Body Dynamics
For complex articulated systems, consider the entire system's dynamics:

```xml
<!-- Example: Humanoid with realistic mass distribution -->
<model name="humanoid_robot">
  <!-- Total mass ~70kg distributed realistically -->
  <link name="base_link">  <!-- Pelvis -->
    <inertial>
      <mass>10.0</mass>  <!-- ~15% of body mass -->
      <inertia>
        <ixx>0.1</ixx>
        <iyy>0.15</iyy>
        <izz>0.2</izz>
      </inertia>
    </inertial>
  </link>

  <!-- Other links with appropriate masses... -->
</model>
```

### Contact Stability
For stable contact with the ground, especially important for bipedal locomotion:

```xml
<!-- Ground plane with high friction for stable walking -->
<model name="ground_plane">
  <static>true</static>
  <link name="link">
    <collision name="collision">
      <geometry>
        <plane>
          <normal>0 0 1</normal>
          <size>100 100</size>
        </plane>
      </geometry>
      <surface>
        <friction>
          <ode>
            <mu>1.0</mu>
            <mu2>1.0</mu2>
          </ode>
        </friction>
        <contact>
          <ode>
            <min_depth>0.001</min_depth>
            <max_vel>100</max_vel>
          </ode>
        </contact>
      </surface>
    </collision>
  </link>
</model>
```

## Performance Optimization

### Simplified Collision Models
Use simplified geometries for collision while keeping detailed visuals:

```xml
<link name="complex_visual_link">
  <!-- Detailed visual for rendering -->
  <visual name="visual">
    <geometry>
      <mesh>
        <uri>file://meshes/detailed_model.dae</uri>
      </mesh>
    </geometry>
  </visual>

  <!-- Simple collision geometry for physics -->
  <collision name="collision">
    <geometry>
      <cylinder>
        <radius>0.1</radius>
        <length>0.5</length>
      </cylinder>
    </geometry>
  </collision>
</link>
```

### Physics Parameter Tuning
Balance accuracy and performance:

```xml
<physics type="ode">
  <!-- For real-time humanoid simulation -->
  <max_step_size>0.002</max_step_size>  <!-- Slightly larger for performance -->
  <real_time_factor>0.8</real_time_factor>  <!-- Allow some slowdown if needed -->
  <real_time_update_rate>500</real_time_update_rate>

  <ode>
    <solver>
      <iters>20</iters>  <!-- Balance between accuracy and speed -->
      <sor>1.3</sor>
    </solver>
  </ode>
</physics>
```

## Troubleshooting Physics Issues

### Common Problems and Solutions

1. **Robot falls through the ground:**
   - Check that collision geometry is properly defined
   - Verify that static models (ground) are actually static
   - Ensure sufficient contact parameters

2. **Unstable or jittery simulation:**
   - Decrease time step size
   - Adjust solver parameters (more iterations)
   - Check mass and inertia properties

3. **Robot tips over easily:**
   - Lower the center of mass
   - Increase base support area
   - Add damping to joints

4. **Slipping feet during walking:**
   - Increase friction coefficients
   - Check contact surface parameters
   - Verify sufficient contact points

## Integration with ROS 2

Physics simulation integrates with ROS 2 through various interfaces:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from geometry_msgs.msg import Wrench
from gazebo_msgs.srv import SetPhysicsProperties
from gazebo_msgs.srv import GetPhysicsProperties


class PhysicsTuner(Node):
    def __init__(self):
        super().__init__('physics_tuner')

        # Service clients for physics control
        self.get_physics_client = self.create_client(
            GetPhysicsProperties, '/gazebo/get_physics_properties')
        self.set_physics_client = self.create_client(
            SetPhysicsProperties, '/gazebo/set_physics_properties')

    def adjust_physics_params(self, time_step, real_time_factor):
        """Dynamically adjust physics parameters during simulation"""
        req = SetPhysicsProperties.Request()
        req.time_step = time_step
        req.real_time_update_rate = 1.0 / time_step
        req.max_update_rate = 0.0
        req.ode_config.sor = 1.3
        req.ode_config.erp = 0.2
        req.ode_config.contact_surface_layer = 0.001

        future = self.set_physics_client.call_async(req)
        # Handle response asynchronously
```

## Summary

Physics simulation in Gazebo requires careful attention to mass properties, collision geometry, and solver parameters. For humanoid robots, realistic physics is essential for developing stable locomotion and manipulation behaviors. The key is finding the right balance between accuracy and performance while ensuring stable contact interactions.

## Learning Check

After completing this section, you should be able to:
- Configure physics engine parameters appropriately
- Calculate and set proper mass and inertia properties
- Define collision geometries for different purposes
- Troubleshoot common physics simulation issues
- Understand the specific requirements for humanoid robot simulation