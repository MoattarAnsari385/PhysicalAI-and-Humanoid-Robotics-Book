# Quickstart Guide: Physical AI & Humanoid Robotics Textbook Development

## Prerequisites

### System Requirements
- **Operating System**: Ubuntu 22.04 LTS (recommended) or equivalent Linux distribution
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 50GB free space for development environment
- **GPU**: NVIDIA GPU with CUDA support recommended for Isaac Sim (RTX 30/40 series or equivalent)

### Software Dependencies
- **Docker**: Version 20.10 or higher
- **Node.js**: Version 18.x or higher
- **ROS 2**: Humble Hawksbill distribution
- **Python**: Version 3.10 or higher
- **Git**: Version 2.25 or higher

## Setup Development Environment

### Option 1: Using Docker (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd Physical-Humanize-Robotic-TextBook

# Build the development container
cd docker
docker build -t robotics-textbook-dev -f Dockerfile .

# Run the container
docker run -it --gpus all -v $(pwd):/workspace robotics-textbook-dev
```

### Option 2: Local Installation
```bash
# Install ROS 2 Humble Hawksbill
sudo apt update && sudo apt install curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install ros-humble-desktop
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-build

# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# Install additional dependencies
pip3 install colcon-common-extensions
pip3 install vcstool
```

## Setting Up the Docusaurus Documentation Site

```bash
# Navigate to the project root
cd /workspace  # or your project directory

# Install Node.js dependencies
npm install

# Start the development server
npm start
```

The documentation site will be available at `http://localhost:3000`.

## Building the Textbook Content

### Content Structure
```
src/
├── modules/             # Content for each module
│   ├── module-1-ros2/   # ROS 2 fundamentals
│   ├── module-2-simulation/  # Gazebo & simulation
│   ├── module-3-ai/     # Isaac Sim & perception
│   └── module-4-vla/    # Vision-Language-Action
├── assets/              # Images, diagrams, and media
├── code/                # Code examples and exercises
└── metadata/            # Glossary and RAG data
```

### Creating New Content
1. Add new markdown files to the appropriate module directory
2. Ensure each section has 3-5 figures and proper alt-text
3. Include code examples with 150-300 words of explanation
4. Add figures to the `assets/` directory with source files

## Running Tests

### Build and Test Documentation
```bash
# Build the site
npm run build

# Run link checker
npm run serve
# In another terminal:
npx linkinator http://localhost:3000 --recurse --verbosity error
```

### Accessibility Testing
```bash
# Install accessibility testing tools
npm install -g pa11y

# Run accessibility tests
pa11y http://localhost:3000 --include-notices --include-warnings
```

### ROS 2 Code Testing
```bash
# Navigate to code directory
cd code/

# Build ROS 2 packages
colcon build

# Run tests
colcon test
colcon test-result --all
```

## Visual Asset Pipeline

### Creating Diagrams
1. Use Inkscape or Diagrams.net for vector diagrams (SVG format)
2. Save source files in `assets/source/`
3. Export optimized versions to `assets/figures/`

### Creating Simulation Renders
1. Use Gazebo 11 or Isaac Sim for high-res renders
2. Include render settings in documentation
3. Save as PNG/JPEG with appropriate resolution

### Regenerating Figures
Scripts for regenerating figures are available in the `scripts/` directory:
```bash
# Run RViz screenshot script
./scripts/rviz-screenshot.sh

# Run Isaac Sim render script
./scripts/isaac-render.sh
```

## Deployment

### GitHub Pages Deployment
The site is automatically deployed via GitHub Actions when changes are pushed to the main branch.

### Manual Deployment
```bash
# Build the site
npm run build

# Deploy to GitHub Pages
GIT_USER=<Your GitHub Username> CURRENT_BRANCH=main USE_SSH=true npm run deploy
```

## Troubleshooting

### Common Issues

1. **ROS 2 Environment Not Found**
   ```bash
   source /opt/ros/humble/setup.bash
   ```

2. **Docusaurus Build Fails**
   - Clear cache: `npm start -- --clear-cache`
   - Clean install: `rm -rf node_modules package-lock.json && npm install`

3. **Simulation Performance Issues**
   - Ensure GPU drivers are properly installed
   - Check CUDA compatibility: `nvidia-smi`

### Getting Help
- Check the detailed documentation in the `docs/` directory
- Review the troubleshooting section in each module
- Join the development community on the project's communication channels