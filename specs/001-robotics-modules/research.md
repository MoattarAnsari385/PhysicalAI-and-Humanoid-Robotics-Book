# Research: Physical AI & Humanoid Robotics Textbook

## Module Structure Decisions

### Decision: 4-module structure with progressive complexity
**Rationale**: Aligns with the user's requirements for a structured learning path from ROS 2 fundamentals to advanced VLA integration. Each module builds upon the previous one, supporting university-level education.

**Alternatives considered**:
- Single comprehensive module (rejected - too overwhelming for students)
- 6+ smaller modules (rejected - might fragment the learning flow)

### Decision: 5-6 sections per module
**Rationale**: Provides sufficient granularity for 6,000-8,000 word modules while maintaining focused learning objectives per section.

## Technology Stack Decisions

### Decision: ROS 2 Humble Hawksbill as primary version
**Rationale**: Long-term support (LTS) release with extensive documentation and community support, ideal for educational content that needs to remain stable over academic terms.

**Alternatives considered**:
- ROS 2 Iron (rejected - newer, less documentation and community support)
- ROS 1 Noetic (rejected - end-of-life, not aligned with industry trends)

### Decision: Gazebo 11 for simulation
**Rationale**: Stable and well-documented simulation environment with good integration with ROS 2. Extensive educational resources available.

**Alternatives considered**:
- Ignition Gazebo (now gz-sim) (rejected - newer, less educational resources)
- Unity (rejected - primarily for game development, different skill set)

### Decision: Python (rclpy) as primary language
**Rationale**: More accessible to students, easier to learn and debug. Better for educational purposes where the focus is on robotics concepts rather than low-level performance.

**Alternatives considered**:
- C++ with rclcpp (rejected - higher learning curve, more complex for beginners)

## Visual Asset Pipeline

### Decision: SVG for diagrams, PNG/JPEG for renders
**Rationale**: SVG provides scalable vector graphics that remain crisp at any resolution, ideal for technical diagrams. Raster formats (PNG/JPEG) are appropriate for simulation renders and real-world images.

**Alternatives considered**:
- All raster formats (rejected - scalability issues for diagrams)
- All vector formats (rejected - not suitable for photographic content)

## Accessibility Standards

### Decision: WCAG 2.1 AA compliance
**Rationale**: This is the standard specified in the clarifications from the specification phase. It provides a good balance between accessibility and implementation effort.

**Alternatives considered**:
- WCAG 2.0 A (rejected - less comprehensive)
- WCAG 2.1 AAA (rejected - excessive implementation effort with minimal benefit)

## Assessment Strategy

### Decision: Formative assessments with automated checking
**Rationale**: Provides continuous feedback to students during their learning process, with automated checking to reduce instructor workload.

**Alternatives considered**:
- Only summative assessments (rejected - less learning feedback)
- Manual-only assessment checking (rejected - too time-intensive)

## Cloud Infrastructure

### Decision: AWS g5.2xlarge as primary cloud option
**Rationale**: Provides A10G GPU with 24GB VRAM, suitable for Isaac Sim and other GPU-intensive robotics tasks. Good cost-performance ratio with spot pricing options.

**Alternatives considered**:
- Azure NV-series (rejected - higher cost for equivalent performance)
- Local RTX hardware (rejected - not accessible to all students)

## Repository Structure

### Decision: Docusaurus-based content organization
**Rationale**: Provides excellent documentation capabilities, search functionality, and accessibility features. Well-suited for educational content.

**Alternatives considered**:
- Jekyll-based static site (rejected - less educational-focused features)
- Custom solution (rejected - higher maintenance overhead)

## Content Density Strategy

### Decision: 800-1,200 words per section with 3-5 figures
**Rationale**: Provides sufficient depth for university-level content while maintaining student engagement. Visual elements break up text and support different learning styles.

**Alternatives considered**:
- Shorter sections (rejected - insufficient depth for complex robotics concepts)
- Longer sections (rejected - potential for cognitive overload)