import math

class RoverAutopilot:
    def __init__(self):
        self.reversing = False
        self.reverse_target_dist = 95.0
        self.target_obstacle = None

    def compute_controls(self, rover, env, dt):
        """
        Compute autonomous forward/backward thrust for rightward exploration.
        Movement is strictly to the right; reversing only engages when necessary
        for hazard/collision avoidance.
        """
        # Slow down/stop if battery depleted
        if rover.battery <= 0.0:
            return 0.0, 0.0

        # Enforce rightward heading (90 deg = East/Right)
        rover.heading = 90.0

        # Scan for immediate hazards in front (ahead to the right)
        safety_distance = 125.0
        collision_distance = 55.0

        closest_obstacle = None
        closest_dist = safety_distance

        for obs in env.rocks + env.craters:
            dx = obs["x"] - rover.x
            # Obstacle must be in front (dx > 0)
            if dx > 0:
                dist = dx - obs.get("r", 20.0)
                # Prioritize unscanned obstacles or imminent collision
                if dist < closest_dist and not obs.get("scanned", False):
                    closest_obstacle = obs
                    closest_dist = dist

        # Check if we should enter or exit reversing mode
        if closest_obstacle is not None and closest_dist < collision_distance:
            self.reversing = True
            self.target_obstacle = closest_obstacle

        if self.reversing:
            # Check if we have backed up to safe distance
            if self.target_obstacle is not None:
                current_dx = self.target_obstacle["x"] - rover.x - self.target_obstacle.get("r", 20.0)
                if current_dx >= self.reverse_target_dist or self.target_obstacle.get("scanned", False):
                    self.reversing = False
                    self.target_obstacle = None
            else:
                self.reversing = False

        # Don't reverse past the landing area boundary
        if rover.x <= rover.min_x + 10.0:
            self.reversing = False

        # Compute forward/backward command
        turn_left_right = 0.0

        if self.reversing:
            # Reversing only when necessary for collision avoidance
            forward_backward = -0.7
            rover.status = "HAZARD AVOIDANCE (REVERSING)"
        elif closest_obstacle is not None:
            # Approaching obstacle: decelerate for stable observation and scanning
            forward_backward = 0.35
            rover.status = "SCANNING HAZARD"
        else:
            # Clear path ahead: full forward exploration to the right
            forward_backward = 1.0
            if rover.status not in ("READY", "EXPLORING"):
                rover.status = "EXPLORING"

        # Boundary check at end of world
        if rover.x > env.world_width - 80.0:
            forward_backward = 0.0
            rover.status = "SURFACE SURVEY COMPLETE"

        return forward_backward, turn_left_right

