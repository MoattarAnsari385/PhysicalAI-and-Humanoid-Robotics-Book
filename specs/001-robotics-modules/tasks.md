---
description: "Task list for Physical AI & Humanoid Robotics Textbook implementation"
---

# Tasks: Physical AI & Humanoid Robotics Textbook

**Input**: Design documents from `/specs/001-robotics-modules/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/, quickstart.md

**Tests**: The examples below include test tasks. Tests are OPTIONAL - only include them if explicitly requested in the feature specification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root
- **Web app**: `backend/src/`, `frontend/src/`
- **Mobile**: `api/src/`, `ios/src/` or `android/src/`
- Paths shown below assume single project - adjust based on plan.md structure

<!--
  ============================================================================
  IMPORTANT: The tasks below are SAMPLE TASKS for illustration purposes only.

  The /sp.tasks command MUST replace these with actual tasks based on:
  - User stories from spec.md (with their priorities P1, P2, P3...)
  - Feature requirements from plan.md
  - Entities from data-model.md
  - Endpoints from contracts/

  Tasks MUST be organized by user story so each story can be:
  - Implemented independently
  - Tested independently
  - Delivered as an MVP increment

  DO NOT keep these sample tasks in the generated tasks.md file.
  ============================================================================
-->

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [X] T001 Create project structure per implementation plan in src/
- [X] T002 [P] Initialize Docusaurus project with Node.js dependencies
- [X] T003 [P] Create docker directory and Dockerfile for development environment
- [X] T004 Create basic directory structure: src/modules/, src/assets/, src/code/, src/metadata/
- [X] T005 [P] Set up GitHub Actions workflow in .github/workflows/main.yml
- [X] T006 Create initial package.json with required dependencies for Docusaurus
- [X] T007 [P] Initialize git repository with appropriate .gitignore for ROS 2/Docusaurus project

---
## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

Examples of foundational tasks (adjust based on your project):

- [X] T008 Create src/modules/module-1-ros2/ directory structure
- [X] T009 [P] Create src/modules/module-2-simulation/ directory structure
- [X] T010 [P] Create src/modules/module-3-ai/ directory structure
- [X] T011 [P] Create src/modules/module-4-vla/ directory structure
- [X] T012 Create src/assets/figures/ and src/assets/source/ directories
- [X] T013 [P] Create src/code/ros-examples/, src/code/gazebo-examples/, src/code/isaac-examples/, src/code/vla-examples/ directories
- [X] T014 Create src/metadata/glossary.json file structure
- [X] T015 Set up Docusaurus configuration for 4-module textbook navigation
- [X] T016 [P] Create basic CI/CD pipeline for build and accessibility checks
- [X] T017 Create scripts directory with figure regeneration utilities

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---
## Phase 3: User Story 1 - University Student Learning Robotics Fundamentals (Priority: P1) 🎯 MVP

**Goal**: Create foundational ROS 2 module covering nodes, topics, services, and actions with a minimal robot package example

**Independent Test**: Students can complete Module 1 (ROS 2) independently and demonstrate understanding of nodes, topics, services, and actions with a basic Python controller.

### Tests for User Story 1 (OPTIONAL - only if tests requested) ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T018 [P] [US1] Create accessibility test script for WCAG 2.1 AA compliance in tests/accessibility/
- [ ] T019 [P] [US1] Set up ROS 2 code testing framework in tests/ros-examples/

### Implementation for User Story 1

- [X] T020 [P] [US1] Create Module 1 metadata file in src/modules/module-1-ros2/module-metadata.json
- [X] T021 [P] [US1] Create 5-6 section markdown files in src/modules/module-1-ros2/ with 800-1200 words each
- [X] T022 [US1] Create basic ROS 2 package example in src/code/ros-examples/minimal-robot-package/
- [X] T023 [P] [US1] Create standard ROS 2 communication diagram in src/assets/figures/ros2-architecture.svg
- [X] T024 [P] [US1] Create URDF for humanoid skeleton in src/code/ros-examples/humanoid-skeleton/
- [X] T025 [US1] Add Python agent controller examples using rclpy in src/code/ros-examples/agent-controllers/
- [ ] T026 [P] [US1] Create 15-18 figures (3-5 per section) for Module 1 in src/assets/figures/
- [X] T027 [US1] Create formative assessments for Module 1 in src/modules/module-1-ros2/assessments/
- [X] T028 [US1] Add alt-text and accessibility features to Module 1 content
- [X] T029 [US1] Create glossary terms for Module 1 in src/metadata/glossary.json
- [X] T030 [US1] Update Docusaurus sidebar to include Module 1 sections

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---
## Phase 4: User Story 2 - Student Learning Simulation and Digital Twin Concepts (Priority: P2)

**Goal**: Create Gazebo simulation module covering dynamics, sensors, and environment interaction with a humanoid test scene

**Independent Test**: Students can complete Module 2 independently and create a functional Gazebo simulation with a humanoid robot navigating a simple environment.

### Tests for User Story 2 (OPTIONAL - only if tests requested) ⚠️

- [X] T031 [P] [US2] Create Gazebo simulation validation tests in tests/gazebo-examples/
- [X] T032 [P] [US2] Set up physics simulation integrity tests

### Implementation for User Story 2

- [X] T033 [P] [US2] Create Module 2 metadata file in src/modules/module-2-simulation/module-metadata.json
- [X] T034 [P] [US2] Create 5-6 section markdown files in src/modules/module-2-simulation/ with 800-1200 words each
- [X] T035 [US2] Create Gazebo world description files in src/code/gazebo-examples/worlds/
- [X] T036 [P] [US2] Create humanoid test scene in Gazebo in src/code/gazebo-examples/humanoid-test-scene/
- [X] T037 [P] [US2] Create physics simulation examples in src/code/gazebo-examples/physics/
- [X] T038 [US2] Create sensor simulation examples (IMU, Depth camera, LiDAR) in src/code/gazebo-examples/sensors/
- [ ] T039 [P] [US2] Create 15-18 figures (3-5 per section) for Module 2 in src/assets/figures/
- [X] T040 [US2] Create formative assessments for Module 2 in src/modules/module-2-simulation/assessments/
- [X] T041 [US2] Add alt-text and accessibility features to Module 2 content
- [X] T042 [US2] Create glossary terms for Module 2 in src/metadata/glossary.json
- [X] T043 [US2] Update Docusaurus sidebar to include Module 2 sections

**Checkpoint**: At this point, User Story 2 should be fully functional and testable independently

---
## Phase 5: User Story 3 - Student Learning AI Perception and Navigation (Priority: P3)

**Goal**: Create NVIDIA Isaac module covering perception, visual SLAM, and navigation with object detection pipeline and navigation workflow

**Independent Test**: Students can complete Module 3 independently and implement an object detection pipeline with navigation planning for a simulated robot.

### Tests for User Story 3 (OPTIONAL - only if tests requested) ⚠️

- [X] T044 [P] [US3] Create Isaac Sim pipeline validation tests in tests/isaac-examples/
- [ ] T045 [P] [US3] Set up perception accuracy tests

### Implementation for User Story 3

- [ ] T046 [P] [US3] Create Module 3 metadata file in src/modules/module-3-ai/module-metadata.json
- [ ] T047 [P] [US3] Create 5-6 section markdown files in src/modules/module-3-ai/ with 800-1200 words each
- [ ] T048 [US3] Create object detection example pipeline in src/code/isaac-examples/object-detection/
- [ ] T049 [P] [US3] Create basic navigation planning workflow in src/code/isaac-examples/navigation/
- [ ] T050 [P] [US3] Create synthetic dataset pipeline examples in src/code/isaac-examples/datasets/
- [ ] T051 [US3] Create Isaac ROS Nav2 A* & RRT navigation examples for biped robots in src/code/isaac-examples/nav2/
- [ ] T052 [P] [US3] Create 15-18 figures (3-5 per section) for Module 3 in src/assets/figures/
- [ ] T053 [US3] Create formative assessments for Module 3 in src/modules/module-3-ai/assessments/
- [ ] T054 [US3] Add alt-text and accessibility features to Module 3 content
- [ ] T055 [US3] Create glossary terms for Module 3 in src/metadata/glossary.json
- [ ] T056 [US3] Update Docusaurus sidebar to include Module 3 sections

**Checkpoint**: At this point, User Story 3 should be fully functional and testable independently

---
## Phase 6: User Story 4 - Student Learning Cognitive Robotics Integration (Priority: P4)

**Goal**: Create VLA module covering LLM-based reasoning with Whisper voice mapping and cognitive planning, culminating in the capstone scenario

**Independent Test**: Students can complete Module 4 independently and demonstrate the complete VLA capstone scenario where a humanoid robot responds to voice commands.

### Tests for User Story 4 (OPTIONAL - only if tests requested) ⚠️

- [X] T057 [P] [US4] Create VLA integration tests in tests/vla-examples/
- [X] T058 [P] [US4] Set up end-to-end capstone scenario validation tests

### Implementation for User Story 4

- [X] T059 [P] [US4] Create Module 4 metadata file in src/modules/module-4-vla/module-metadata.json
- [X] T060 [P] [US4] Create 5-6 section markdown files in src/modules/module-4-vla/ with 800-1200 words each
- [X] T061 [US4] Create Whisper voice to ROS 2 command mapping examples in src/code/vla-examples/voice-mapping/
- [X] T062 [P] [US4] Create cognitive planning to action sequence examples in src/code/vla-examples/planning/
- [X] T063 [P] [US4] Create task planning flowchart in src/assets/figures/task-planning-flowchart.svg
- [X] T064 [US4] Create VLA integration architecture diagram in src/assets/figures/vla-architecture.svg
- [X] T065 [P] [US4] Implement capstone scenario: "Humanoid receives voice input → creates plan → navigates obstacles → identifies object → manipulates object" in src/code/vla-examples/capstone/
- [X] T066 [P] [US4] Create 15-18 figures (3-5 per section) for Module 4 in src/assets/figures/
- [X] T067 [US4] Create formative assessments for Module 4 in src/modules/module-4-vla/assessments/
- [X] T068 [US4] Add alt-text and accessibility features to Module 4 content
- [X] T069 [US4] Create glossary terms for Module 4 in src/metadata/glossary.json
- [X] T070 [US4] Update Docusaurus sidebar to include Module 4 sections

**Checkpoint**: All user stories should now be independently functional

---
[Add more user story phases as needed, following the same pattern]

---
## Phase N: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T071 [P] Documentation updates and cross-module references in docs/
- [ ] T072 Code cleanup and consistency across all modules
- [ ] T073 [P] Performance optimization for Docusaurus site build time
- [ ] T074 [P] Accessibility validation across all modules using pa11y
- [ ] T075 Link checking across all content using linkinator
- [ ] T076 [P] Final glossary compilation and cross-referencing in src/metadata/glossary.json
- [ ] T077 Comprehensive testing of all ROS 2 examples using colcon
- [ ] T078 [P] Final review of content against learning outcomes
- [ ] T079 Run quickstart.md validation across all modules

---
## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3 → P4)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - May integrate with US1 but should be independently testable
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - May integrate with US1/US2 but should be independently testable
- **User Story 4 (P4)**: Can start after Foundational (Phase 2) - Builds on all previous stories for capstone integration

### Within Each User Story

- Tests (if included) MUST be written and FAIL before implementation
- Models before services
- Services before endpoints
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- All tests for a user story marked [P] can run in parallel
- Models within a story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members

---
## Parallel Example: User Story 1

```bash
# Launch all setup tasks for User Story 1 together:
T020 [P] [US1] Create Module 1 metadata file in src/modules/module-1-ros2/module-metadata.json
T021 [P] [US1] Create 5-6 section markdown files in src/modules/module-1-ros2/ with 800-1200 words each
T023 [P] [US1] Create standard ROS 2 communication diagram in src/assets/figures/ros2-architecture.svg
T024 [P] [US1] Create URDF for humanoid skeleton in src/code/ros-examples/humanoid-skeleton/
T026 [P] [US1] Create 15-18 figures (3-5 per section) for Module 1 in src/assets/figures/
```

---
## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Add User Story 4 → Test independently → Deploy/Demo
6. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Story 2
   - Developer C: User Story 3
   - Developer D: User Story 4
3. Stories complete and integrate independently

---
## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence