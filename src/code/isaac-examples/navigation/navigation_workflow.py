"""
Navigation Planning Workflow for NVIDIA Isaac
Task: T049 [P] [US3] Create basic navigation planning workflow in src/code/isaac-examples/navigation/

This module implements a navigation planning workflow using NVIDIA Isaac Sim and Isaac ROS.
It demonstrates path planning, obstacle avoidance, and navigation execution for humanoid robots.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import math


class NavigationState(Enum):
    """Navigation state enumeration"""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    ADJUSTING = "adjusting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Pose2D:
    """2D pose representation"""
    x: float
    y: float
    theta: float  # orientation in radians


@dataclass
class Path:
    """Navigation path representation"""
    waypoints: List[Pose2D]
    cost: float
    valid: bool


@dataclass
class NavigationResult:
    """Navigation execution result"""
    success: bool
    path: Path
    execution_time: float
    distance_traveled: float
    final_pose: Pose2D


class GridMap:
    """Simple 2D grid map for navigation planning"""

    def __init__(self, width: int, height: int, resolution: float = 1.0):
        """
        Initialize the grid map

        Args:
            width: Map width in grid cells
            height: Map height in grid cells
            resolution: Size of each grid cell in meters
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        self.grid = np.zeros((height, width), dtype=np.uint8)  # 0: free, 1: occupied

    def set_obstacle(self, x: int, y: int):
        """Set a cell as occupied (obstacle)"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] = 1

    def is_free(self, x: int, y: int) -> bool:
        """Check if a cell is free of obstacles"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y, x] == 0
        return False

    def world_to_grid(self, x_world: float, y_world: float) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates"""
        x_grid = int(x_world / self.resolution)
        y_grid = int(y_world / self.resolution)
        return x_grid, y_grid

    def grid_to_world(self, x_grid: int, y_grid: int) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates"""
        x_world = x_grid * self.resolution
        y_world = y_grid * self.resolution
        return x_world, y_world


class PathPlanner:
    """Path planning algorithm implementation"""

    def __init__(self, grid_map: GridMap):
        self.grid_map = grid_map

    def plan_path(self, start: Pose2D, goal: Pose2D) -> Path:
        """
        Plan a path from start to goal using A* algorithm

        Args:
            start: Starting pose
            goal: Goal pose

        Returns:
            Planned path
        """
        # Convert world coordinates to grid coordinates
        start_grid = self.grid_map.world_to_grid(start.x, start.y)
        goal_grid = self.grid_map.world_to_grid(goal.x, goal.y)

        # Check if start and goal are in valid positions
        if not self.grid_map.is_free(*start_grid) or not self.grid_map.is_free(*goal_grid):
            return Path([], float('inf'), False)

        # A* pathfinding implementation
        open_set = [(start_grid, 0 + self._heuristic(start_grid, goal_grid))]
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self._heuristic(start_grid, goal_grid)}

        while open_set:
            # Find node with lowest f_score
            current = min(open_set, key=lambda x: x[1])[0]
            open_set = [item for item in open_set if item[0] != current]

            if current == goal_grid:
                # Reconstruct path
                path = self._reconstruct_path(came_from, current)
                return Path(path, g_score[current], True)

            # Check neighbors
            for neighbor in self._get_neighbors(current):
                if not self.grid_map.is_free(*neighbor):
                    continue

                tentative_g_score = g_score.get(current, float('inf')) + self._distance(current, neighbor)

                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, goal_grid)
                    open_set.append((neighbor, f_score[neighbor]))

        # No path found
        return Path([], float('inf'), False)

    def _heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate heuristic (Euclidean distance) between two positions"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighbors for a position"""
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]:
            new_x, new_y = pos[0] + dx, pos[1] + dy
            if 0 <= new_x < self.grid_map.width and 0 <= new_y < self.grid_map.height:
                neighbors.append((new_x, new_y))
        return neighbors

    def _distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate distance between two positions"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def _reconstruct_path(self, came_from: Dict, current: Tuple[int, int]) -> List[Pose2D]:
        """Reconstruct path from came_from dictionary"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)

        # Convert grid coordinates back to world coordinates and reverse the path
        world_path = []
        for grid_pos in reversed(path):
            x, y = self.grid_map.grid_to_world(grid_pos[0], grid_pos[1])
            world_path.append(Pose2D(x, y, 0.0))  # Assuming zero orientation for path points

        return world_path


class LocalPlanner:
    """Local path planning for dynamic obstacle avoidance"""

    def __init__(self, local_map_size: int = 10):
        self.local_map_size = local_map_size

    def plan_local_path(self, current_pose: Pose2D, global_path: Path,
                       local_obstacles: List[Tuple[float, float]]) -> Path:
        """
        Plan a local path considering dynamic obstacles

        Args:
            current_pose: Current robot pose
            global_path: Global path to follow
            local_obstacles: List of local obstacles in world coordinates

        Returns:
            Local path adjustment
        """
        # For this example, we'll return a simplified local adjustment
        # In a real implementation, this would use dynamic window approach or similar
        if local_obstacles:
            # Adjust path to avoid obstacles
            adjusted_waypoints = []
            for waypoint in global_path.waypoints:
                # Check if waypoint is near any obstacle
                safe = True
                for obs_x, obs_y in local_obstacles:
                    dist = math.sqrt((waypoint.x - obs_x)**2 + (waypoint.y - obs_y)**2)
                    if dist < 1.0:  # 1 meter safety margin
                        safe = False
                        break

                if safe:
                    adjusted_waypoints.append(waypoint)
                else:
                    # Find alternative route around obstacle
                    adjusted_waypoints.append(Pose2D(
                        waypoint.x + 0.5,  # Slightly offset
                        waypoint.y + 0.5,
                        waypoint.theta
                    ))

            return Path(adjusted_waypoints, global_path.cost, True)
        else:
            return global_path


class NavigationController:
    """Navigation controller for executing planned paths"""

    def __init__(self, linear_vel: float = 0.5, angular_vel: float = 0.5):
        """
        Initialize the navigation controller

        Args:
            linear_vel: Maximum linear velocity (m/s)
            angular_vel: Maximum angular velocity (rad/s)
        """
        self.linear_vel = linear_vel
        self.angular_vel = angular_vel
        self.current_state = NavigationState.IDLE

    def execute_path(self, path: Path, current_pose: Pose2D) -> NavigationResult:
        """
        Execute a planned path

        Args:
            path: Path to execute
            current_pose: Current robot pose

        Returns:
            Navigation execution result
        """
        if not path.valid or len(path.waypoints) == 0:
            return NavigationResult(False, path, 0.0, 0.0, current_pose)

        # Simulate path execution
        import time
        start_time = time.time()
        distance_traveled = 0.0
        current_pos = current_pose

        for i, waypoint in enumerate(path.waypoints):
            # Calculate distance to waypoint
            dist_to_waypoint = math.sqrt(
                (waypoint.x - current_pos.x)**2 +
                (waypoint.y - current_pos.y)**2
            )
            distance_traveled += dist_to_waypoint

            # Update current position (simulation)
            current_pos = waypoint

            # Simulate execution time based on distance
            time.sleep(0.01)  # Simulate processing time

        execution_time = time.time() - start_time
        final_pose = path.waypoints[-1] if path.waypoints else current_pose

        return NavigationResult(True, path, execution_time, distance_traveled, final_pose)


class IsaacNavigationSystem:
    """Complete Isaac navigation system integrating planning and execution"""

    def __init__(self, map_width: int = 50, map_height: int = 50, resolution: float = 0.5):
        """
        Initialize the Isaac navigation system

        Args:
            map_width: Width of the navigation map (in grid cells)
            map_height: Height of the navigation map (in grid cells)
            resolution: Size of each grid cell (in meters)
        """
        self.grid_map = GridMap(map_width, map_height, resolution)
        self.path_planner = PathPlanner(self.grid_map)
        self.local_planner = LocalPlanner()
        self.controller = NavigationController()
        self.current_state = NavigationState.IDLE

    def add_obstacle(self, x: float, y: float):
        """Add an obstacle to the navigation map"""
        grid_x, grid_y = self.grid_map.world_to_grid(x, y)
        self.grid_map.set_obstacle(grid_x, grid_y)

    def navigate_to_goal(self, start: Pose2D, goal: Pose2D) -> NavigationResult:
        """
        Navigate from start to goal

        Args:
            start: Starting pose
            goal: Goal pose

        Returns:
            Navigation result
        """
        self.current_state = NavigationState.PLANNING

        # Plan global path
        global_path = self.path_planner.plan_path(start, goal)
        if not global_path.valid:
            self.current_state = NavigationState.FAILED
            return NavigationResult(False, global_path, 0.0, 0.0, start)

        self.current_state = NavigationState.EXECUTING

        # Execute the path
        result = self.controller.execute_path(global_path, start)
        self.current_state = NavigationState.COMPLETED if result.success else NavigationState.FAILED

        return result

    def navigate_with_dynamic_obstacles(self, start: Pose2D, goal: Pose2D,
                                      dynamic_obstacles: List[Tuple[float, float]]) -> NavigationResult:
        """
        Navigate with consideration of dynamic obstacles

        Args:
            start: Starting pose
            goal: Goal pose
            dynamic_obstacles: List of dynamic obstacles [x, y]

        Returns:
            Navigation result
        """
        self.current_state = NavigationState.PLANNING

        # Plan initial global path
        global_path = self.path_planner.plan_path(start, goal)
        if not global_path.valid:
            self.current_state = NavigationState.FAILED
            return NavigationResult(False, global_path, 0.0, 0.0, start)

        # Adjust for dynamic obstacles
        local_path = self.local_planner.plan_local_path(start, global_path, dynamic_obstacles)

        self.current_state = NavigationState.EXECUTING

        # Execute the adjusted path
        result = self.controller.execute_path(local_path, start)
        self.current_state = NavigationState.COMPLETED if result.success else NavigationState.FAILED

        return result


def main():
    """Example usage of the navigation system"""
    print("Initializing Isaac Navigation System...")

    # Create navigation system
    nav_system = IsaacNavigationSystem()

    # Add some static obstacles to the map
    for x, y in [(10, 10), (11, 10), (12, 10), (10, 11), (10, 12)]:  # Wall
        nav_system.add_obstacle(x, y)

    for x, y in [(20, 15), (21, 15), (20, 16)]:  # Small obstacle
        nav_system.add_obstacle(x, y)

    # Define start and goal positions
    start = Pose2D(5.0, 5.0, 0.0)
    goal = Pose2D(30.0, 25.0, 0.0)

    print(f"Planning path from {start} to {goal}...")
    result = nav_system.navigate_to_goal(start, goal)

    if result.success:
        print(f"Navigation completed successfully!")
        print(f"Execution time: {result.execution_time:.2f} seconds")
        print(f"Distance traveled: {result.distance_traveled:.2f} meters")
        print(f"Final pose: {result.final_pose}")
    else:
        print("Navigation failed!")

    # Demonstrate navigation with dynamic obstacles
    print("\nTesting navigation with dynamic obstacles...")
    dynamic_obs = [(25.0, 20.0), (26.0, 21.0)]  # Moving obstacles
    result2 = nav_system.navigate_with_dynamic_obstacles(start, goal, dynamic_obs)

    if result2.success:
        print(f"Dynamic navigation completed successfully!")
        print(f"Execution time: {result2.execution_time:.2f} seconds")
        print(f"Distance traveled: {result2.distance_traveled:.2f} meters")
    else:
        print("Dynamic navigation failed!")


if __name__ == "__main__":
    main()