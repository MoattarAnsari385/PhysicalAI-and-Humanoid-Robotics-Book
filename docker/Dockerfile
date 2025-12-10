# Use ROS 2 Humble Hawksbill as the base image
FROM osrf/ros:humble-desktop

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js and npm for Docusaurus
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Create workspace directory
WORKDIR /workspace

# Copy package files
COPY package*.json ./

# Install Node.js dependencies
RUN npm install

# Install ROS 2 Python dependencies
RUN pip3 install --upgrade pip \
    && pip3 install rclpy geometry_msgs sensor_msgs nav_msgs

# Set up rosdep
RUN rosdep init || true
RUN rosdep update

# Create source directory and copy example package
RUN mkdir -p src
COPY src/code/ros-examples/minimal-robot-package ./src/minimal_robot_package

# Build the workspace
RUN source /opt/ros/$ROS_DISTRO/setup.bash && \
    colcon build --packages-select minimal_robot_package

# Source ROS 2 environment
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> ~/.bashrc
RUN echo "source /workspace/install/setup.bash" >> ~/.bashrc

# Expose port for Docusaurus
EXPOSE 3000

# Set the default command
CMD ["bash"]