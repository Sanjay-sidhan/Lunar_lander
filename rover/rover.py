import math
import random

class LunarRover:
    def __init__(self, start_x=260, start_y=1500):
        self.start_x = start_x
        self.start_y = start_y
        self.min_x = 100.0  # Landing site left boundary
        self.reset()

    def reset(self):
        self.x = float(self.start_x)
        self.y = float(self.start_y)
        self.speed = 0.0          # m/s (speed magnitude)
        self.heading = 90.0       # Strict forward heading: 90 is East/Right
        self.battery = 100.0      # percentage
        self.temperature = -71.0  # Celsius
        self.distance_travelled = 0.0  # meters
        self.status = "READY"     # READY, EXPLORING, REVERSING (HAZARD AVOIDANCE), STANDBY
        self.scanned_rocks = 0
        self.scanned_craters = 0
        self.mode = "AUTONOMOUS"  # AUTONOMOUS or MANUAL
        self.trail = []

    def update(self, dt):
        """Update rover status, thermodynamics, battery, and travel telemetry."""
        # Simulated temperature fluctuations
        self.temperature = -71.0 + random.uniform(-1.5, 1.5)

        # Add to trail if moved
        if not self.trail or math.hypot(self.x - self.trail[-1][0], self.y - self.trail[-1][1]) > 8.0:
            self.trail.append((self.x, self.y))
            if len(self.trail) > 200:
                self.trail.pop(0)

        if abs(self.speed) > 0.01:
            # Drain battery relative to speed
            self.battery = max(0.0, self.battery - 0.5 * abs(self.speed) * dt)
            
            # Calculate distance increment
            dist = abs(self.speed) * dt * 10.0  # Scale speed for visual progression
            self.distance_travelled += dist

            # Update status
            if self.battery <= 0.0:
                self.speed = 0.0
                self.status = "STANDBY (BATTERY DEPLETED)"
            elif self.speed < -0.05:
                self.status = "HAZARD AVOIDANCE (REVERSING)"
            else:
                self.status = "EXPLORING"
        else:
            if self.battery <= 0.0:
                self.status = "STANDBY (BATTERY DEPLETED)"
            elif self.status not in ("HAZARD_STOPPED", "READY", "HAZARD AVOIDANCE (REVERSING)"):
                self.status = "READY"

    def move(self, forward_backward, turn_left_right, dt):
        """
        Adjust heading and speed based on control inputs.
        forward_backward: +1 (forward), -1 (backward), 0 (none)
        turn_left_right: -1 (left turn), +1 (right turn), 0 (none)
        """
        if self.battery <= 0.0:
            return

        # Adjust heading
        if turn_left_right != 0:
            self.heading = (self.heading + turn_left_right * 90.0 * dt) % 360.0

        # Adjust speed (accel / decelerate)
        max_speed = 2.5
        accel = 1.5
        drag = 1.0

        if forward_backward > 0:
            self.speed = min(max_speed, self.speed + accel * dt)
        elif forward_backward < 0:
            self.speed = max(-max_speed, self.speed - accel * dt)
        else:
            # Apply drag to stop rover
            if self.speed > 0:
                self.speed = max(0.0, self.speed - drag * dt)
            elif self.speed < 0:
                self.speed = min(0.0, self.speed + drag * dt)

        # Move coordinates based on heading and speed
        # Convert heading to radians (0 deg is UP, so dx = sin(heading), dy = -cos(heading))
        rad = math.radians(self.heading)
        dx = math.sin(rad) * self.speed * dt * 40.0  # scaling factor for Pygame coordinates
        dy = -math.cos(rad) * self.speed * dt * 40.0
        
        self.x += dx
        self.y += dy

        # Enforce left boundary (landing pad area)
        if self.x < self.min_x:
            self.x = self.min_x
            if (self.heading == 90.0 and self.speed < 0) or (self.heading == 270.0 and self.speed > 0):
                self.speed = 0.0

    def get_telemetry(self):
        return {
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "speed": round(self.speed, 2),
            "battery": round(self.battery, 1),
            "temperature": round(self.temperature, 1),
            "heading": round(self.heading, 1),
            "distance": round(self.distance_travelled, 1),
            "status": self.status,
            "mode": self.mode,
            "rocks": self.scanned_rocks,
            "craters": self.scanned_craters
        }
