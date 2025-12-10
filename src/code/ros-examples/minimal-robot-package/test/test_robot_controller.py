# Copyright 2023 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import rclpy
from minimal_robot_package.robot_controller import RobotController


class TestRobotController(unittest.TestCase):

    def setUp(self):
        rclpy.init()
        self.node = RobotController()

    def tearDown(self):
        self.node.destroy_node()
        rclpy.shutdown()

    def test_node_creation(self):
        """Test that the robot controller node was created successfully"""
        self.assertIsNotNone(self.node)
        self.assertEqual(self.node.get_name(), 'robot_controller')

    def test_subscribers_created(self):
        """Test that required subscribers were created"""
        # Check that subscriptions exist (they're stored as attributes)
        self.assertIsNotNone(self.node.scan_sub)
        self.assertIsNotNone(self.node.odom_sub)

    def test_publishers_created(self):
        """Test that required publishers were created"""
        # Check that publishers exist
        self.assertIsNotNone(self.node.cmd_vel_pub)


if __name__ == '__main__':
    unittest.main()