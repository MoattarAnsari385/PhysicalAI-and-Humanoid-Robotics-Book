"""
Isaac ROS Nav2 Navigation for Biped Robots
Task: T051 [US3] Create Isaac ROS Nav2 A* & RRT navigation examples for biped robots in src/code/isaac-examples/nav2/

This module implements navigation algorithms (A* and RRT) specifically adapted for biped robots
using Isaac ROS and Nav2. It demonstrates how to handle the unique challenges of bipedal navigation.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import math
import heapq
from scipy.spatial import KDTree


class RobotType(Enum):
    """Type of robot for navigation planning"""
    BIPED = "biped"
    WHEELED = "wheeled"
    HOVER = "hover"


@dataclass
class BipedState:
    """State representation for biped robot"""
    x: float
    y: float
    theta: float  # orientation
    left_foot_x: float
    left_foot_y: float
    right_foot_x: float
    right_foot_y: float
    step_phase: float  # 0.0 to 1.0, where 0.5 is double support phase


@dataclass
class PathNode:
    """Node in path planning"""
    state: BipedState
    cost: float
    heuristic: float
    parent: Optional['PathNode'] = None

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)


class BipedKinematics:
    """Kinematic model for biped robots"""

    def __init__(self, leg_length: float = 0.8, step_length: float = 0.6, step_height: float = 0.1):
        """
        Initialize biped kinematics model

        Args:
            leg_length: Length of robot's legs
            step_length: Maximum length of a single step
            step_height: Maximum height of a step
        """
        self.leg_length = leg_length
        self.step_length = step_length
        self.step_height = step_height
        self.step_width = 0.3  # Distance between feet

    def is_state_valid(self, state: BipedState, obstacles: np.ndarray) -> bool:
        """
        Check if a biped state is valid (no collisions)

        Args:
            state: Biped robot state to check
            obstacles: Array of obstacle positions [(x, y), ...]

        Returns:
            True if state is valid, False otherwise
        """
        # Check if robot body collides with obstacles
        robot_points = [
            (state.x, state.y),  # Body center
            (state.left_foot_x, state.left_foot_y),  # Left foot
            (state.right_foot_x, state.right_foot_y)  # Right foot
        ]

        for point in robot_points:
            for obs_x, obs_y in obstacles:
                dist = math.sqrt((point[0] - obs_x)**2 + (point[1] - obs_y)**2)
                if dist < 0.3:  # Collision threshold
                    return False

        return True

    def get_next_state(self, current_state: BipedState, step_type: str, step_size: float = 0.3) -> Optional[BipedState]:
        """
        Calculate next state based on step type

        Args:
            current_state: Current robot state
            step_type: Type of step ('left_forward', 'right_forward', 'left_backward', 'right_backward', 'turn_left', 'turn_right')
            step_size: Size of the step

        Returns:
            Next state or None if invalid
        """
        new_state = BipedState(
            x=current_state.x,
            y=current_state.y,
            theta=current_state.theta,
            left_foot_x=current_state.left_foot_x,
            left_foot_y=current_state.left_foot_y,
            right_foot_x=current_state.right_foot_x,
            right_foot_y=current_state.right_foot_y,
            step_phase=0.0
        )

        # Calculate step offset based on current orientation
        dx = step_size * math.cos(current_state.theta)
        dy = step_size * math.sin(current_state.theta)
        perp_dx = step_size * math.cos(current_state.theta + math.pi/2)
        perp_dy = step_size * math.sin(current_state.theta + math.pi/2)

        if step_type == 'left_forward':
            new_state.left_foot_x += dx
            new_state.left_foot_y += dy
            new_state.x += dx / 2  # Body moves with the step
            new_state.y += dy / 2
        elif step_type == 'right_forward':
            new_state.right_foot_x += dx
            new_state.right_foot_y += dy
            new_state.x += dx / 2
            new_state.y += dy / 2
        elif step_type == 'left_backward':
            new_state.left_foot_x -= dx
            new_state.left_foot_y -= dy
            new_state.x -= dx / 2
            new_state.y -= dy / 2
        elif step_type == 'right_backward':
            new_state.right_foot_x -= dx
            new_state.right_foot_y -= dy
            new_state.x -= dx / 2
            new_state.y -= dy / 2
        elif step_type == 'turn_left':
            new_state.theta += 0.2  # 0.2 radian turn
            # Move feet to maintain balance during turn
            new_state.left_foot_x += perp_dx * 0.5
            new_state.left_foot_y += perp_dy * 0.5
            new_state.right_foot_x -= perp_dx * 0.5
            new_state.right_foot_y -= perp_dy * 0.5
        elif step_type == 'turn_right':
            new_state.theta -= 0.2  # 0.2 radian turn
            # Move feet to maintain balance during turn
            new_state.left_foot_x -= perp_dx * 0.5
            new_state.left_foot_y -= perp_dy * 0.5
            new_state.right_foot_x += perp_dx * 0.5
            new_state.right_foot_y += perp_dy * 0.5
        else:
            return None

        return new_state


class AStarBipedPlanner:
    """A* path planner adapted for biped robots"""

    def __init__(self, grid_map: np.ndarray, resolution: float = 1.0):
        """
        Initialize A* planner for biped navigation

        Args:
            grid_map: 2D grid map (0: free, 1: occupied)
            resolution: Resolution of the grid in meters
        """
        self.grid_map = grid_map
        self.resolution = resolution
        self.height, self.width = grid_map.shape
        self.kinematics = BipedKinematics()

    def plan_path(self, start_state: BipedState, goal_pos: Tuple[float, float],
                  obstacles: np.ndarray) -> Optional[List[BipedState]]:
        """
        Plan path using A* algorithm adapted for biped robots

        Args:
            start_state: Starting state of the biped robot
            goal_pos: Goal position (x, y)
            obstacles: Array of obstacle positions

        Returns:
            List of states forming the path, or None if no path found
        """
        # Convert positions to grid coordinates
        start_grid = self._world_to_grid(start_state.x, start_state.y)
        goal_grid = self._world_to_grid(goal_pos[0], goal_pos[1])

        # Priority queue for A*
        open_set = []
        heapq.heappush(open_set, (0, id(start_state), start_state))

        # Cost dictionaries
        g_score = {id(start_state): 0}
        f_score = {id(start_state): self._heuristic(start_state, goal_pos)}

        # Path reconstruction
        came_from = {}

        while open_set:
            current = heapq.heappop(open_set)[2]

            # Check if we've reached the goal area
            dist_to_goal = math.sqrt((current.x - goal_pos[0])**2 + (current.y - goal_pos[1])**2)
            if dist_to_goal < 1.0:  # Within 1 meter of goal
                return self._reconstruct_path(came_from, current)

            # Generate possible next states (different step types)
            step_types = ['left_forward', 'right_forward', 'left_backward', 'right_backward',
                         'turn_left', 'turn_right']

            for step_type in step_types:
                next_state = self.kinematics.get_next_state(current, step_type)

                if next_state is None:
                    continue

                # Check if the new state is valid
                if not self.kinematics.is_state_valid(next_state, obstacles):
                    continue

                # Calculate tentative g_score
                step_cost = self._step_cost(current, next_state)
                tentative_g_score = g_score[id(current)] + step_cost

                next_state_id = id(next_state)
                if next_state_id not in g_score or tentative_g_score < g_score[next_state_id]:
                    came_from[next_state_id] = current
                    g_score[next_state_id] = tentative_g_score
                    f_score[next_state_id] = tentative_g_score + self._heuristic(next_state, goal_pos)
                    heapq.heappush(open_set, (f_score[next_state_id], next_state_id, next_state))

        return None  # No path found

    def _heuristic(self, state: BipedState, goal: Tuple[float, float]) -> float:
        """Calculate heuristic (Euclidean distance)"""
        return math.sqrt((state.x - goal[0])**2 + (state.y - goal[1])**2)

    def _step_cost(self, from_state: BipedState, to_state: BipedState) -> float:
        """Calculate cost of a step"""
        # Base cost is the distance moved
        dist_cost = math.sqrt((to_state.x - from_state.x)**2 + (to_state.y - from_state.y)**2)

        # Add penalty for turning
        turn_cost = abs(to_state.theta - from_state.theta) * 0.5

        # Add penalty for unbalanced states
        balance_cost = self._balance_penalty(to_state) * 0.2

        return dist_cost + turn_cost + balance_cost

    def _balance_penalty(self, state: BipedState) -> float:
        """Calculate penalty for unbalanced biped state"""
        # Calculate distance between feet
        foot_distance = math.sqrt(
            (state.left_foot_x - state.right_foot_x)**2 +
            (state.left_foot_y - state.right_foot_y)**2
        )

        # Optimal foot distance is around step_width
        optimal_distance = self.kinematics.step_width
        distance_diff = abs(foot_distance - optimal_distance)

        # Penalty increases as feet get too close or too far
        return max(0, distance_diff - 0.1)  # Small tolerance

    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates"""
        grid_x = int(x / self.resolution)
        grid_y = int(y / self.resolution)
        return min(grid_x, self.width - 1), min(grid_y, self.height - 1)

    def _reconstruct_path(self, came_from: Dict, current: BipedState) -> List[BipedState]:
        """Reconstruct path from came_from dictionary"""
        path = [current]
        current_id = id(current)

        while current_id in came_from:
            current = came_from[current_id]
            path.append(current)
            current_id = id(current)

        return list(reversed(path))


class RRTPebbedPlanner:
    """RRT planner adapted for biped robots"""

    def __init__(self, grid_map: np.ndarray, resolution: float = 1.0,
                 max_iterations: int = 1000, step_size: float = 0.5):
        """
        Initialize RRT planner for biped navigation

        Args:
            grid_map: 2D grid map (0: free, 1: occupied)
            resolution: Resolution of the grid in meters
            max_iterations: Maximum number of RRT iterations
            step_size: Step size for RRT extension
        """
        self.grid_map = grid_map
        self.resolution = resolution
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.kinematics = BipedKinematics()

    def plan_path(self, start_state: BipedState, goal_pos: Tuple[float, float],
                  obstacles: np.ndarray) -> Optional[List[BipedState]]:
        """
        Plan path using RRT algorithm adapted for biped robots

        Args:
            start_state: Starting state of the biped robot
            goal_pos: Goal position (x, y)
            obstacles: Array of obstacle positions

        Returns:
            List of states forming the path, or None if no path found
        """
        # Tree of states
        tree = [start_state]
        state_to_idx = {id(start_state): 0}

        for iteration in range(self.max_iterations):
            # Sample random state or bias toward goal
            if iteration % 10 == 0:  # Bias toward goal every 10 iterations
                random_state = self._create_random_state_near_goal(goal_pos)
            else:
                random_state = self._create_random_state()

            # Find nearest state in tree
            nearest_idx = self._nearest_state_idx(tree, random_state)
            nearest_state = tree[nearest_idx]

            # Extend toward random state
            new_state = self._extend_toward(nearest_state, random_state, obstacles)

            if new_state is not None:
                tree.append(new_state)
                state_to_idx[id(new_state)] = len(tree) - 1

                # Check if we're close to the goal
                dist_to_goal = math.sqrt((new_state.x - goal_pos[0])**2 + (new_state.y - goal_pos[1])**2)
                if dist_to_goal < 1.0:
                    return self._extract_path(tree, state_to_idx, start_state, new_state)

        return None  # No path found

    def _create_random_state(self) -> BipedState:
        """Create a random state within map bounds"""
        x = np.random.uniform(0, self.grid_map.shape[1] * self.resolution)
        y = np.random.uniform(0, self.grid_map.shape[0] * self.resolution)
        theta = np.random.uniform(0, 2 * math.pi)

        # Simple foot positioning based on body position
        left_foot_x = x + 0.15 * math.cos(theta + math.pi/2)
        left_foot_y = y + 0.15 * math.sin(theta + math.pi/2)
        right_foot_x = x + 0.15 * math.cos(theta - math.pi/2)
        right_foot_y = y + 0.15 * math.sin(theta - math.pi/2)

        return BipedState(x, y, theta, left_foot_x, left_foot_y, right_foot_x, right_foot_y, 0.0)

    def _create_random_state_near_goal(self, goal_pos: Tuple[float, float]) -> BipedState:
        """Create a random state near the goal"""
        # Add some randomness around the goal
        x = goal_pos[0] + np.random.uniform(-1.0, 1.0)
        y = goal_pos[1] + np.random.uniform(-1.0, 1.0)
        theta = np.random.uniform(0, 2 * math.pi)

        # Simple foot positioning based on body position
        left_foot_x = x + 0.15 * math.cos(theta + math.pi/2)
        left_foot_y = y + 0.15 * math.sin(theta + math.pi/2)
        right_foot_x = x + 0.15 * math.cos(theta - math.pi/2)
        right_foot_y = y + 0.15 * math.sin(theta - math.pi/2)

        return BipedState(x, y, theta, left_foot_x, left_foot_y, right_foot_x, right_foot_y, 0.0)

    def _nearest_state_idx(self, tree: List[BipedState], target: BipedState) -> int:
        """Find index of nearest state in tree to target"""
        min_dist = float('inf')
        nearest_idx = 0

        for i, state in enumerate(tree):
            dist = math.sqrt((state.x - target.x)**2 + (state.y - target.y)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        return nearest_idx

    def _extend_toward(self, from_state: BipedState, to_state: BipedState,
                      obstacles: np.ndarray) -> Optional[BipedState]:
        """Extend from_state toward to_state"""
        # Calculate direction vector
        dx = to_state.x - from_state.x
        dy = to_state.y - from_state.y
        dist = math.sqrt(dx**2 + dy**2)

        if dist < self.step_size:
            candidate_state = to_state
        else:
            # Scale the step to step_size
            scale = self.step_size / dist
            target_x = from_state.x + dx * scale
            target_y = from_state.y + dy * scale

            # Create a candidate state with the new position
            candidate_state = BipedState(
                x=target_x,
                y=target_y,
                theta=to_state.theta,  # Maintain target orientation
                left_foot_x=to_state.left_foot_x,
                left_foot_y=to_state.left_foot_y,
                right_foot_x=to_state.right_foot_x,
                right_foot_y=to_state.right_foot_y,
                step_phase=0.0
            )

        # Check if the candidate state is valid
        if self.kinematics.is_state_valid(candidate_state, obstacles):
            return candidate_state
        else:
            return None

    def _extract_path(self, tree: List[BipedState], state_to_idx: Dict,
                     start_state: BipedState, goal_state: BipedState) -> List[BipedState]:
        """Extract path from start to goal through the tree"""
        # For simplicity, return just the start and goal for now
        # In a full implementation, we would trace back through the tree
        return [start_state, goal_state]


class IsaacBipedNavigationSystem:
    """Complete navigation system for biped robots using Isaac ROS and Nav2"""

    def __init__(self, map_width: int = 50, map_height: int = 50, resolution: float = 0.5):
        """
        Initialize the biped navigation system

        Args:
            map_width: Width of the navigation map
            map_height: Height of the navigation map
            resolution: Resolution of the map in meters
        """
        self.resolution = resolution
        self.grid_map = np.zeros((map_height, map_width))
        self.obstacles = np.array([]).reshape(0, 2)  # Empty obstacle array initially
        self.astar_planner = None
        self.rrt_planner = None

    def add_obstacle(self, x: float, y: float, radius: float = 0.5):
        """Add a circular obstacle to the map"""
        # Add to list of obstacles
        self.obstacles = np.append(self.obstacles, [[x, y]], axis=0)

        # Update grid map
        grid_x = int(x / self.resolution)
        grid_y = int(y / self.resolution)
        grid_radius = int(radius / self.resolution)

        for dx in range(-grid_radius, grid_radius + 1):
            for dy in range(-grid_radius, grid_radius + 1):
                if dx*dx + dy*dy <= grid_radius*grid_radius:
                    new_x, new_y = grid_x + dx, grid_y + dy
                    if 0 <= new_x < self.grid_map.shape[1] and 0 <= new_y < self.grid_map.shape[0]:
                        self.grid_map[new_y, new_x] = 1

    def plan_path_astar(self, start_pos: Tuple[float, float], goal_pos: Tuple[float, float]) -> Optional[List[BipedState]]:
        """
        Plan path using A* algorithm

        Args:
            start_pos: Starting position (x, y)
            goal_pos: Goal position (x, y)

        Returns:
            Path as list of BipedState, or None if no path found
        """
        # Initialize A* planner
        self.astar_planner = AStarBipedPlanner(self.grid_map, self.resolution)

        # Create initial biped state
        start_state = BipedState(
            x=start_pos[0],
            y=start_pos[1],
            theta=0.0,
            left_foot_x=start_pos[0] + 0.15,  # Offset feet slightly
            left_foot_y=start_pos[1],
            right_foot_x=start_pos[0] - 0.15,
            right_foot_y=start_pos[1],
            step_phase=0.0
        )

        # Plan path
        return self.astar_planner.plan_path(start_state, goal_pos, self.obstacles)

    def plan_path_rrt(self, start_pos: Tuple[float, float], goal_pos: Tuple[float, float]) -> Optional[List[BipedState]]:
        """
        Plan path using RRT algorithm

        Args:
            start_pos: Starting position (x, y)
            goal_pos: Goal position (x, y)

        Returns:
            Path as list of BipedState, or None if no path found
        """
        # Initialize RRT planner
        self.rrt_planner = RRTPebbedPlanner(self.grid_map, self.resolution)

        # Create initial biped state
        start_state = BipedState(
            x=start_pos[0],
            y=start_pos[1],
            theta=0.0,
            left_foot_x=start_pos[0] + 0.15,  # Offset feet slightly
            left_foot_y=start_pos[1],
            right_foot_x=start_pos[0] - 0.15,
            right_foot_y=start_pos[1],
            step_phase=0.0
        )

        # Plan path
        return self.rrt_planner.plan_path(start_state, goal_pos, self.obstacles)

    def execute_path(self, path: List[BipedState]) -> bool:
        """
        Execute the planned path (simulation)

        Args:
            path: Path to execute

        Returns:
            True if execution successful
        """
        if not path:
            return False

        print(f"Executing path with {len(path)} waypoints...")

        # Simulate path execution
        for i, state in enumerate(path):
            print(f"Moving to waypoint {i+1}/{len(path)}: ({state.x:.2f}, {state.y:.2f})")
            # In a real implementation, this would send commands to the robot

        print("Path execution completed!")
        return True


def main():
    """Example usage of the biped navigation system"""
    print("Initializing Isaac Biped Navigation System...")

    # Create navigation system
    nav_system = IsaacBipedNavigationSystem()

    # Add some obstacles to the map
    nav_system.add_obstacle(10.0, 10.0, 1.0)
    nav_system.add_obstacle(15.0, 15.0, 0.8)
    nav_system.add_obstacle(20.0, 8.0, 1.2)
    nav_system.add_obstacle(25.0, 20.0, 0.9)

    # Define start and goal positions
    start_pos = (5.0, 5.0)
    goal_pos = (40.0, 40.0)

    print(f"Planning path from {start_pos} to {goal_pos} using A*...")

    # Plan path using A*
    astar_path = nav_system.plan_path_astar(start_pos, goal_pos)

    if astar_path:
        print(f"A* found path with {len(astar_path)} waypoints")
        # Execute A* path
        nav_system.execute_path(astar_path)
    else:
        print("A* failed to find a path")

    print(f"\nPlanning path from {start_pos} to {goal_pos} using RRT...")

    # Plan path using RRT
    rrt_path = nav_system.plan_path_rrt(start_pos, goal_pos)

    if rrt_path:
        print(f"RRT found path with {len(rrt_path)} waypoints")
        # Execute RRT path
        nav_system.execute_path(rrt_path)
    else:
        print("RRT failed to find a path")


if __name__ == "__main__":
    main()