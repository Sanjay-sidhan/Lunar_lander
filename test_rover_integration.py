import os
import sys
import unittest
import json
import time

# Add base directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mission.mission_controller import MissionController, MissionState
from rover.rover import LunarRover
from rover.rover_env import RoverEnv
from rover.rover_navigation import RoverAutopilot
from rover.rover_telemetry import RoverTelemetryLogger

class TestRoverIntegration(unittest.TestCase):
    def test_mission_controller_transitions(self):
        mc = MissionController()
        self.assertEqual(mc.state, MissionState.LANDING)

        # Transition to landed
        mc.change_state(MissionState.LANDED)
        self.assertEqual(mc.state, MissionState.LANDED)
        
        # Test timer update to transition state
        mc.update(3.0)
        self.assertEqual(mc.state, MissionState.TRANSITION)

        # Test transition update to deployment
        mc.update(1.6)
        self.assertEqual(mc.state, MissionState.ROVER_DEPLOYMENT)

        # Test deployment updates to exploration
        mc.update(4.0)
        self.assertEqual(mc.state, MissionState.ROVER_EXPLORATION)

    def test_rover_physics_and_telemetry(self):
        rover = LunarRover(start_x=100, start_y=100)
        self.assertEqual(rover.x, 100.0)
        self.assertEqual(rover.y, 100.0)
        self.assertEqual(rover.speed, 0.0)

        # Move forward manually
        rover.move(1.0, 0.0, 0.1)  # forward_backward = 1.0, dt = 0.1
        rover.update(0.1)
        self.assertTrue(rover.speed > 0.0)
        self.assertTrue(rover.distance_travelled > 0.0)
        
        telemetry = rover.get_telemetry()
        self.assertEqual(telemetry["status"], "EXPLORING")
        self.assertTrue(telemetry["distance"] > 0.0)

    def test_rover_env_generation(self):
        env = RoverEnv()
        self.assertTrue(len(env.rocks) > 0)
        self.assertTrue(len(env.craters) > 0)
        
        rover = LunarRover(start_x=1500, start_y=1500)
        terrain_type = env.get_terrain_type(rover)
        self.assertIsInstance(terrain_type, str)

        hz_score = env.get_hazard_score(rover)
        self.assertGreaterEqual(hz_score, 0.0)
        self.assertLessEqual(hz_score, 1.0)

    def test_rover_telemetry_logger(self):
        test_file = "test_telemetry.json"
        if os.path.exists(test_file):
            os.remove(test_file)

        logger = RoverTelemetryLogger(filepath=test_file)
        rover = LunarRover(start_x=100, start_y=100)
        
        logger.add_record(rover.get_telemetry())
        
        # Manually trigger write by adjusting last_write_time
        logger.last_write_time = time.time() - 2.0
        logger.write_to_disk()

        self.assertTrue(os.path.exists(test_file))
        with open(test_file, 'r') as f:
            data = json.load(f)
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["x"], 100.0)

        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)

    def test_rover_movement_and_hazard_reversing(self):
        rover = LunarRover(start_x=260, start_y=1500)
        env = RoverEnv()
        autopilot = RoverAutopilot()

        self.assertEqual(rover.heading, 90.0)
        self.assertEqual(rover.x, 260.0)

        # Clear path: autopilot moves right (forward_backward > 0)
        fb, lr = autopilot.compute_controls(rover, env, 0.1)
        self.assertEqual(rover.heading, 90.0)
        self.assertGreater(fb, 0.0)

        # Place an obstacle directly in front within collision distance
        test_rock = {"x": rover.x + 30.0, "y": 1500.0, "r": 20.0, "scanned": False}
        env.rocks.insert(0, test_rock)

        # Autopilot should command reverse (fb < 0) without turning around (heading remains 90.0)
        fb, lr = autopilot.compute_controls(rover, env, 0.1)
        self.assertLess(fb, 0.0, "Rover should reverse when obstacle is dangerously close")
        self.assertEqual(rover.heading, 90.0, "Rover heading must remain 90.0 (rightward)")

        # Rover moves in reverse
        rover.move(fb, lr, 0.1)
        rover.update(0.1)
        self.assertEqual(rover.status, "HAZARD AVOIDANCE (REVERSING)")

    def test_camera_frame_optical_capture(self):
        rover = LunarRover(start_x=260, start_y=1500)
        env = RoverEnv()
        frame = env.get_camera_frame(rover, width=200, height=120)
        
        self.assertIsNotNone(frame)
        self.assertEqual(frame.get_size(), (200, 120))

    def test_manual_mode_move_left(self):
        rover = LunarRover(start_x=300, start_y=1500)
        rover.mode = "MANUAL"
        rover.heading = 270.0  # Facing left
        
        initial_x = rover.x
        # Move forward while facing left
        rover.move(1.0, 0.0, 0.1)
        rover.update(0.1)
        
        self.assertLess(rover.x, initial_x, "Rover must move left (decrease X) when heading is 270.0")


if __name__ == "__main__":
    unittest.main()
