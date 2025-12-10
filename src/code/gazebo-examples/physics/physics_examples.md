# Physics Simulation Examples

This document provides examples of different physics configurations for humanoid robots in Gazebo.

## Basic Physics Configuration

The most common physics setup for humanoid robots balances accuracy with performance:

```xml
<physics type="ode">
  <max_step_size>0.002</max_step_size>  <!-- 2ms time step -->
  <real_time_factor>0.8</real_time_factor>  <!-- Allow slight slowdown -->
  <real_time_update_rate>500</real_time_update_rate>
  <gravity>0 0 -9.8</gravity>
  <ode>
    <solver>
      <type>quick</type>
      <iters>20</iters>  <!-- More iterations for stability -->
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0.000001</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

## High-Accuracy Physics Configuration

For precise simulation of complex interactions:

```xml
<physics type="ode">
  <max_step_size>0.001</max_step_size>  <!-- Smaller time step for accuracy -->
  <real_time_factor>0.5</real_time_factor>  <!-- Lower RTF for accuracy -->
  <real_time_update_rate>1000</real_time_update_rate>
  <gravity>0 0 -9.8</gravity>
  <ode>
    <solver>
      <type>quick</type>
      <iters>50</iters>  <!-- More iterations for precision -->
      <sor>1.0</sor>
    </solver>
    <constraints>
      <cfm>0.0000001</cfm>  <!-- Lower CFM for tighter constraints -->
      <erp>0.1</erp>  <!-- Lower ERP for less error -->
      <contact_max_correcting_vel>10</contact_max_correcting_vel>
      <contact_surface_layer>0.0001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

## Performance-Oriented Physics Configuration

For faster simulation when accuracy is less critical:

```xml
<physics type="ode">
  <max_step_size>0.01</max_step_size>  <!-- Larger time step -->
  <real_time_factor>1.5</real_time_factor>  <!-- Higher RTF -->
  <real_time_update_rate>100</real_time_update_rate>
  <gravity>0 0 -9.8</gravity>
  <ode>
    <solver>
      <type>quick</type>
      <iters>10</iters>  <!-- Fewer iterations -->
      <sor>1.5</sor>
    </solver>
    <constraints>
      <cfm>0.00001</cfm>
      <erp>0.5</erp>
      <contact_max_correcting_vel>1000</contact_max_correcting_vel>
      <contact_surface_layer>0.01</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

## Link Inertial Properties Examples

Proper inertial properties are critical for realistic physics:

### Simple Box Link
```xml
<inertial>
  <mass>1.0</mass>
  <pose>0 0 0 0 0 0</pose>
  <inertia>
    <ixx>0.083333</ixx>  <!-- For 1x1x1 box: m*(h²+d²)/12 -->
    <ixy>0</ixy>
    <ixz>0</ixz>
    <iyy>0.083333</iyy>  <!-- For 1x1x1 box: m*(w²+d²)/12 -->
    <iyz>0</iyz>
    <izz>0.083333</izz>  <!-- For 1x1x1 box: m*(w²+h²)/12 -->
  </inertia>
</inertial>
```

### Cylinder Link
```xml
<inertial>
  <mass>0.5</mass>
  <pose>0 0 0 0 0 0</pose>
  <inertia>
    <ixx>0.005208</ixx>  <!-- mr²/4 + mh²/12 (about x-axis) -->
    <ixy>0</ixy>
    <ixz>0</ixz>
    <iyy>0.005208</iyy>  <!-- mr²/4 + mh²/12 (about y-axis) -->
    <iyz>0</iyz>
    <izz>0.0025</izz>    <!-- mr²/2 (about z-axis) -->
  </inertia>
</inertial>
```

### Humanoid Limb (Approximated as Capsule)
```xml
<inertial>
  <mass>0.8</mass>
  <pose>0 0 -0.2 0 0 0</pose>  <!-- COM offset toward middle -->
  <inertia>
    <ixx>0.0107</ixx>  <!-- For capsule: m*(r²/4 + h²/12) -->
    <ixy>0</ixy>
    <ixz>0.008</ixz>   <!-- Off-diagonal term due to COM offset -->
    <iyy>0.0107</iyy>
    <iyz>0</iyz>
    <izz>0.004</izz>   <!-- For capsule: m*r²/2 -->
  </inertia>
</inertial>
```

## Contact Properties Examples

### High-Friction Surface (Good for Walking)
```xml
<surface>
  <friction>
    <ode>
      <mu>1.0</mu>      <!-- Static friction coefficient -->
      <mu2>1.0</mu2>    <!-- Secondary friction coefficient -->
      <fdir1>0 0 1</fdir1>  <!-- Friction direction (optional) -->
    </ode>
  </friction>
  <contact>
    <ode>
      <min_depth>0.001</min_depth>  <!-- Penetration tolerance -->
      <max_vel>100</max_vel>        <!-- Max correction velocity -->
    </ode>
  </contact>
  <bounce>
    <restitution_coefficient>0.1</restitution_coefficient>  <!-- Bounciness -->
    <threshold>100000</threshold>  <!-- Velocity threshold for bounce -->
  </bounce>
</surface>
```

### Low-Friction Surface (Challenging for Walking)
```xml
<surface>
  <friction>
    <ode>
      <mu>0.1</mu>    <!-- Very low friction -->
      <mu2>0.1</mu2>
    </ode>
  </friction>
  <contact>
    <ode>
      <min_depth>0.002</min_depth>  <!-- Slightly higher penetration -->
      <max_vel>100</max_vel>
    </ode>
  </contact>
</surface>
```

## Damping Examples

Damping helps stabilize simulations and model energy loss:

### Link with Damping
```xml
<link name="damped_link">
  <inertial>
    <mass>1.0</mass>
    <inertia>
      <ixx>0.1</ixx>
      <iyy>0.1</iyy>
      <izz>0.1</izz>
    </inertia>
    <linear_damping>0.1</linear_damping>    <!-- Air resistance -->
    <angular_damping>0.1</angular_damping>  <!-- Rotational resistance -->
  </inertial>
</link>
```

### Joint with Damping
```xml
<joint name="damped_joint" type="revolute">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  <dynamics damping="0.5" friction="0.1"/>  <!-- Joint-specific damping -->
</joint>
```

## Complete Humanoid Physics Example

Here's a complete example of a simple humanoid with proper physics:

```xml
<?xml version="1.0" ?>
<robot name="simple_humanoid">
  <!-- Pelvis/Body -->
  <link name="base_link">
    <inertial>
      <mass>10.0</mass>
      <inertia>
        <ixx>0.5</ixx>
        <iyy>0.8</iyy>
        <izz>0.6</izz>
      </inertia>
    </inertial>
    <visual>
      <geometry>
        <box size="0.3 0.2 0.4"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.4"/>
      </geometry>
    </collision>
  </link>

  <!-- Head -->
  <link name="head">
    <inertial>
      <mass>2.0</mass>
      <inertia>
        <ixx>0.02</ixx>
        <iyy>0.02</iyy>
        <izz>0.02</izz>
      </inertia>
    </inertial>
    <visual>
      <geometry>
        <sphere radius="0.12"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.12"/>
      </geometry>
    </collision>
  </link>

  <joint name="neck" type="fixed">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0 0 0.3"/>
  </joint>

  <!-- Left Upper Leg -->
  <link name="left_upper_leg">
    <inertial>
      <mass>1.5</mass>
      <inertia>
        <ixx>0.05</ixx>
        <iyy>0.05</iyy>
        <izz>0.01</izz>
      </inertia>
    </inertial>
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_hip" type="revolute">
    <parent link="base_link"/>
    <child link="left_upper_leg"/>
    <origin xyz="0.08 0 -0.2"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="2"/>
    <dynamics damping="0.5" friction="0.1"/>
  </joint>

  <!-- Left Lower Leg -->
  <link name="left_lower_leg">
    <inertial>
      <mass>1.0</mass>
      <inertia>
        <ixx>0.03</ixx>
        <iyy>0.03</iyy>
        <izz>0.008</izz>
      </inertia>
    </inertial>
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_knee" type="revolute">
    <parent link="left_upper_leg"/>
    <child link="left_lower_leg"/>
    <origin xyz="0 0 -0.4"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.35" effort="100" velocity="2"/>
    <dynamics damping="0.5" friction="0.1"/>
  </joint>

  <!-- Left Foot -->
  <link name="left_foot">
    <inertial>
      <mass>0.5</mass>
      <inertia>
        <ixx>0.005</ixx>
        <iyy>0.01</iyy>
        <izz>0.008</izz>
      </inertia>
    </inertial>
    <visual>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_ankle" type="fixed">
    <parent link="left_lower_leg"/>
    <child link="left_foot"/>
    <origin xyz="0 0 -0.2"/>
  </joint>

  <!-- Similar for right leg, arms, etc. -->
</robot>
```

## Troubleshooting Physics Issues

### Common Problems and Solutions:

1. **Robot falls through ground**: Check collision geometry, ensure ground is static, verify gravity direction

2. **Unstable simulation**: Reduce time step, increase solver iterations, check mass/inertia values

3. **Jittery movement**: Increase ERP, decrease CFM, add damping

4. **Excessive sliding**: Increase friction coefficients, check contact parameters

5. **Joint limits not respected**: Verify joint type and limit values, check for conflicting constraints

These examples provide a foundation for configuring physics properties for humanoid robots in Gazebo. Remember to tune parameters based on your specific robot design and simulation requirements.