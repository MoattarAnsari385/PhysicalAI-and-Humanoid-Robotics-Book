# Sensor Simulation Examples

This document provides examples of different sensor configurations for humanoid robots in Gazebo.

## Camera Sensor Example

A basic RGB camera for visual perception:

```xml
<sensor name="camera" type="camera">
  <always_on>true</always_on>
  <update_rate>30</update_rate>
  <camera name="head_camera">
    <horizontal_fov>1.047</horizontal_fov>  <!-- 60 degrees in radians -->
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>10.0</far>
    </clip>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.007</stddev>
    </noise>
  </camera>
  <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
    <frame_name>camera_optical_frame</frame_name>
    <topic_name>camera/image_raw</topic_name>
    <camera_info_topic_name>camera/camera_info</camera_info_topic_name>
    <hack_baseline>0.07</hack_baseline>
  </plugin>
</sensor>
```

## Depth Camera Sensor Example

A depth camera that provides both RGB and depth information:

```xml
<sensor name="depth_camera" type="depth">
  <always_on>true</always_on>
  <update_rate>30</update_rate>
  <camera name="head_depth_camera">
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <depth_camera>
      <output>log</output>
    </depth_camera>
    <clip>
      <near>0.1</near>
      <far>10.0</far>
    </clip>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.01</stddev>
    </noise>
  </camera>
  <plugin name="depth_camera_controller" filename="libgazebo_ros_openni_kinect.so">
    <baseline>0.2</baseline>
    <distortion_k1>0.0</distortion_k1>
    <distortion_k2>0.0</distortion_k2>
    <distortion_k3>0.0</distortion_k3>
    <distortion_t1>0.0</distortion_t1>
    <distortion_t2>0.0</distortion_t2>
    <point_cloud_cutoff>0.3</point_cloud_cutoff>
    <point_cloud_cutoff_max>3.0</point_cloud_cutoff_max>
    <frame_name>depth_camera_optical_frame</frame_name>
    <min_depth>0.1</min_depth>
    <max_depth>10.0</max_depth>
    <always_on>true</always_on>
    <update_rate>30.0</update_rate>
    <topic_name>depth_camera/depth/image_raw</topic_name>
    <depth_image_topic_name>depth_camera/depth/image_rect_raw</depth_image_topic_name>
    <point_cloud_topic_name>depth_camera/depth/points</point_cloud_topic_name>
    <camera_info_topic_name>depth_camera/camera_info</camera_info_topic_name>
  </plugin>
</sensor>
```

## LiDAR Sensor Example (2D)

A 2D LiDAR for navigation and obstacle detection:

```xml
<sensor name="laser_2d" type="ray">
  <always_on>true</always_on>
  <update_rate>10</update_rate>
  <ray>
    <scan>
      <horizontal>
        <samples>720</samples>      <!-- Number of rays per revolution -->
        <resolution>1</resolution>   <!-- Angular resolution -->
        <min_angle>-3.14159</min_angle>  <!-- -180 degrees -->
        <max_angle>3.14159</max_angle>    <!-- 180 degrees -->
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>    <!-- Minimum detectable range -->
      <max>30.0</max>   <!-- Maximum detectable range -->
      <resolution>0.01</resolution>  <!-- Range resolution -->
    </range>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.01</stddev>
    </noise>
  </ray>
  <plugin name="laser_2d_controller" filename="libgazebo_ros_ray_sensor.so">
    <ros>
      <namespace>/robot</namespace>
      <remapping>~/out:=scan</remapping>
    </ros>
    <output_type>sensor_msgs/LaserScan</output_type>
    <frame_name>laser_frame</frame_name>
    <topic_name>scan</topic_name>
  </plugin>
</sensor>
```

## 3D LiDAR Sensor Example (Velodyne-style)

A multi-line LiDAR for 3D mapping:

```xml
<sensor name="velodyne_lidar" type="ray">
  <always_on>true</always_on>
  <update_rate>10</update_rate>
  <ray>
    <scan>
      <horizontal>
        <samples>1800</samples>
        <resolution>1</resolution>
        <min_angle>-3.14159</min_angle>
        <max_angle>3.14159</max_angle>
      </horizontal>
      <vertical>
        <samples>16</samples>
        <resolution>1</resolution>
        <min_angle>-0.261799</min_angle>  <!-- -15 degrees -->
        <max_angle>0.261799</max_angle>    <!-- 15 degrees -->
      </vertical>
    </scan>
    <range>
      <min>0.2</min>
      <max>100.0</max>
      <resolution>0.001</resolution>
    </range>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.008</stddev>
    </noise>
  </ray>
  <plugin name="velodyne_controller" filename="libgazebo_ros_velodyne_gpu_laser.so">
    <topic_name>velodyne_points</topic_name>
    <frame_name>velodyne</frame_name>
    <min_range>0.2</min_range>
    <max_range>100.0</max_range>
    <gaussian_noise>0.008</gaussian_noise>
  </plugin>
</sensor>
```

## IMU Sensor Example

An Inertial Measurement Unit for orientation and acceleration:

```xml
<sensor name="imu_sensor" type="imu">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.0000075</bias_mean>
          <bias_stddev>0.0000008</bias_stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.0000075</bias_mean>
          <bias_stddev>0.0000008</bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.0000075</bias_mean>
          <bias_stddev>0.0000008</bias_stddev>
        </noise>
      </z>
    </angular_velocity>
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
  <plugin name="imu_controller" filename="libgazebo_ros_imu.so">
    <frame_name>imu_link</frame_name>
    <topic_name>imu/data</topic_name>
    <serviceName>imu/service</serviceName>
    <gaussianNoise>0.01</gaussianNoise>
  </plugin>
</sensor>
```

## Force/Torque Sensor Example

A force/torque sensor for detecting contact forces:

```xml
<sensor name="force_torque_sensor" type="force_torque">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <force_torque>
    <frame>sensor</frame>
    <measure_direction>child_to_parent</measure_direction>
  </force_torque>
  <plugin name="ft_controller" filename="libgazebo_ros_ft.so">
    <frame_name>ft_sensor_frame</frame_name>
    <topic_name>wrench</topic_name>
    <update_rate>100.0</update_rate>
  </plugin>
</sensor>
```

## GPS Sensor Example

A GPS sensor for outdoor localization:

```xml
<sensor name="gps_sensor" type="gps">
  <always_on>true</always_on>
  <update_rate>10</update_rate>
  <plugin name="gps_controller" filename="libgazebo_ros_gps.so">
    <frame_name>gps_frame</frame_name>
    <topic_name>fix</topic_name>
    <update_rate>10.0</update_rate>
    <gaussian_noise>0.02</gaussian_noise>
  </plugin>
</sensor>
```

## Sensor Placement on Humanoid Robot

Here's an example of how to place different sensors on a humanoid robot:

```xml
<link name="head">
  <!-- Front-facing camera -->
  <sensor name="front_camera" type="camera">
    <pose>0.05 0 0 0 0 0</pose>  <!-- Positioned at front of head -->
    <!-- ... camera configuration ... -->
  </sensor>

  <!-- Depth camera -->
  <sensor name="depth_camera" type="depth">
    <pose>0.05 0 0.05 0 0 0</pose>  <!-- Slightly above camera -->
    <!-- ... depth camera configuration ... -->
  </sensor>

  <!-- IMU in head for orientation -->
  <sensor name="head_imu" type="imu">
    <pose>0 0 0.05 0 0 0</pose>  <!-- Center of head -->
    <!-- ... IMU configuration ... -->
  </sensor>
</link>

<link name="torso">
  <!-- Main IMU for body orientation -->
  <sensor name="torso_imu" type="imu">
    <pose>0 0 0.1 0 0 0</pose>  <!-- Center of torso -->
    <!-- ... IMU configuration ... -->
  </sensor>

  <!-- 2D LiDAR mounted on chest -->
  <sensor name="chest_lidar" type="ray">
    <pose>0.1 0 0 0 0 0</pose>  <!-- Front of torso -->
    <!-- ... LiDAR configuration ... -->
  </sensor>
</link>

<link name="left_foot">
  <!-- Force/torque sensors for ground contact detection -->
  <sensor name="left_foot_ft" type="force_torque">
    <pose>0 0 -0.05 0 0 0</pose>  <!-- Bottom of foot -->
    <!-- ... force/torque configuration ... -->
  </sensor>
</link>

<link name="right_foot">
  <sensor name="right_foot_ft" type="force_torque">
    <pose>0 0 -0.05 0 0 0</pose>
    <!-- ... force/torque configuration ... -->
  </sensor>
</link>
```

## Multi-Sensor Fusion Example

Combining multiple sensors for enhanced perception:

```xml
<!-- Stereo camera setup -->
<link name="stereo_mount">
  <!-- Left camera -->
  <sensor name="left_camera" type="camera">
    <pose>-0.05 0.07 0 0 0 0</pose>  <!-- 7cm right of center -->
    <camera name="left_eye">
      <horizontal_fov>1.047</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>10.0</far>
      </clip>
    </camera>
    <plugin name="left_camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>left_camera_optical_frame</frame_name>
      <topic_name>stereo/left/image_raw</topic_name>
      <camera_info_topic_name>stereo/left/camera_info</camera_info_topic_name>
    </plugin>
  </sensor>

  <!-- Right camera -->
  <sensor name="right_camera" type="camera">
    <pose>-0.05 -0.07 0 0 0 0</pose>  <!-- 7cm left of center -->
    <camera name="right_eye">
      <horizontal_fov>1.047</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>10.0</far>
      </clip>
    </camera>
    <plugin name="right_camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>right_camera_optical_frame</frame_name>
      <topic_name>stereo/right/image_raw</topic_name>
      <camera_info_topic_name>stereo/right/camera_info</camera_info_topic_name>
    </plugin>
  </sensor>
</link>
```

## Sensor Noise Modeling

Realistic sensor noise modeling is important for robust algorithm development:

### Camera Noise
```xml
<camera name="noisy_camera">
  <image>
    <width>640</width>
    <height>480</height>
    <format>R8G8B8</format>
  </image>
  <noise>
    <type>gaussian</type>
    <mean>0.0</mean>
    <stddev>0.007</stddev>  <!-- Add some Gaussian noise -->
  </noise>
</camera>
```

### LiDAR Noise
```xml
<ray>
  <range>
    <min>0.1</min>
    <max>30.0</max>
    <resolution>0.01</resolution>
  </range>
  <noise>
    <type>gaussian</type>
    <mean>0.0</mean>
    <stddev>0.01</stddev>  <!-- 1cm noise -->
  </noise>
</ray>
```

## Performance Considerations

### Update Rates
Different sensors have different computational requirements:

```xml
<!-- High-rate sensors (100Hz) -->
<sensor name="imu" type="imu">
  <update_rate>100</update_rate>
  <!-- ... configuration ... -->
</sensor>

<!-- Medium-rate sensors (30Hz) -->
<sensor name="camera" type="camera">
  <update_rate>30</update_rate>
  <!-- ... configuration ... -->
</sensor>

<!-- Low-rate sensors (10Hz) -->
<sensor name="lidar" type="ray">
  <update_rate>10</update_rate>
  <!-- ... configuration ... -->
</sensor>
```

## Sensor Validation and Testing

### Basic Sensor Test Node
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, LaserScan
from cv_bridge import CvBridge
import numpy as np

class SensorValidator(Node):
    def __init__(self):
        super().__init__('sensor_validator')

        self.cv_bridge = CvBridge()

        # Subscribe to different sensor types
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

        self.get_logger().info('Sensor validator started')

    def image_callback(self, msg):
        """Process and validate camera data"""
        try:
            # Convert to OpenCV image
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Validate image properties
            height, width, channels = cv_image.shape

            # Log validation results
            if width == 640 and height == 480:
                self.get_logger().info(f'Camera OK: {width}x{height}, {channels} channels')
            else:
                self.get_logger().warn(f'Unexpected image size: {width}x{height}')

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def imu_callback(self, msg):
        """Process and validate IMU data"""
        # Check orientation quaternion validity
        quat = msg.orientation
        norm = (quat.x**2 + quat.y**2 + quat.z**2 + quat.w**2)**0.5

        if abs(norm - 1.0) > 0.1:
            self.get_logger().warn(f'Invalid quaternion norm: {norm}')
        else:
            self.get_logger().info(f'IMU OK: quaternion norm = {norm:.3f}')

    def scan_callback(self, msg):
        """Process and validate LiDAR data"""
        # Check for valid range values
        valid_ranges = [r for r in msg.ranges if r >= msg.range_min and r <= msg.range_max and not np.isnan(r)]

        if len(valid_ranges) == len(msg.ranges):
            self.get_logger().info(f'LiDAR OK: {len(msg.ranges)} valid ranges')
        else:
            invalid_count = len(msg.ranges) - len(valid_ranges)
            self.get_logger().warn(f'LiDAR: {invalid_count} invalid ranges out of {len(msg.ranges)}')

def main(args=None):
    rclpy.init(args=args)
    validator = SensorValidator()

    try:
        rclpy.spin(validator)
    except KeyboardInterrupt:
        pass
    finally:
        validator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Troubleshooting Common Sensor Issues

### 1. Sensor Not Publishing Data
- Check that the sensor plugin is properly loaded
- Verify topic names and namespaces
- Ensure the sensor is enabled (`always_on` set to true)

### 2. Noisy Sensor Data
- Verify noise parameters match real sensor specifications
- Check physics update rates and time steps
- Ensure proper grounding of sensor links

### 3. Incorrect Sensor Orientation
- Verify sensor poses relative to parent links
- Check coordinate frame conventions (ROS vs. Gazebo)
- Ensure proper TF tree setup

### 4. Performance Issues
- Reduce sensor update rates
- Lower resolution for cameras
- Simplify sensor models if possible

These examples provide a comprehensive foundation for implementing various sensor types on humanoid robots in Gazebo. Remember to tune parameters based on your specific robot design and simulation requirements.