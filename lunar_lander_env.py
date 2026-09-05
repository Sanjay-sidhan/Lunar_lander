"""
lunar_lander_env.py
====================
Gym-compatible environment wrapper for the Lunar Lander RL agent (PPO).

Observation Space (8 values, all normalized to [-1, 1]):
    [lander_x, lander_y, vx, vy, angle, fuel, on_pad, near_ground]

Action Space (Discrete, 6 actions):
    0 - Do nothing
    1 - Main thrust (up)
    2 - Left drift
    3 - Right drift
    4 - Rotate left
    5 - Rotate right

FIXES APPLIED:
    - Bug 1: angle no longer hard-coded to 0 on landing — always returns real angle
    - Bug 2: speed is now the true raw magnitude (units/step), not a fake scaled display value
    - Bug 4: fuel efficiency (steps survived per unit fuel burned) tracked and returned in info
"""

import math
import random
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces


# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────
WIDTH, HEIGHT   = 1200, 700
GROUND_Y        = HEIGHT - 110
PAD_HALF_WIDTH  = 100
MAX_FUEL        = 100.0
MAX_STEPS       = 1200
GRAVITY         = 0.05
THRUST_POWER    = 0.12
FUEL_COST       = 0.3
LATERAL_POWER   = 0.03
ROTATE_SPEED    = 2.0

RW_LAND         =  250.0   # big positive for clean landing
RW_CRASH        = -150.0   # reduced crash penalty so agent isn't too risk-averse
RW_FUEL_STEP    = -0.02    # tiny per-thrust penalty (was -0.05, too harsh)
RW_VY_SCALE     = -1.2     # penalise fast downward speed near ground strongly
RW_VX_SCALE     = -0.8     # penalise lateral drift near ground
RW_ANGLE_SCALE  = -0.6     # penalise tilt near ground
RW_STEP         = -0.01    # very small time penalty (was -0.05, discouraged exploring)

# --- NEW positive shaping constants ---
RW_PAD_APPROACH =  0.8     # reward for moving closer to pad (potential-based)
RW_FUEL_BONUS   =  1.5     # reward per unit of fuel remaining at landing
RW_SOFT_LAND    =  50.0    # bonus for very gentle touchdown (vy < 0.5)
RW_UPRIGHT      =  20.0    # bonus for landing nearly upright (angle < 5°)


def _clip(val, lo, hi):
    return max(lo, min(hi, val))


# ──────────────────────────────────────────────
# ENVIRONMENT
# ──────────────────────────────────────────────
class LunarLanderEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, randomize_pad=False, wind=False):
        super().__init__()

        self.render_mode   = render_mode
        self.randomize_pad = randomize_pad
        self.wind_enabled  = wind

        low  = np.array([-1, -1, -1, -1, -1,  0, 0, 0], dtype=np.float32)
        high = np.array([ 1,  1,  1,  1,  1,  1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space      = spaces.Discrete(6)

        self._state   = {}
        self._steps   = 0
        self._prev_dist_to_pad = None   # for potential-based approach reward

        # FIX 4: track cumulative fuel burned per episode for efficiency metric
        self._fuel_used = 0.0

        # ── Pad position ───────────────────────────────────────────────
        self._pinned_pad_x = None
        self._pad_x1 = WIDTH  // 2 - PAD_HALF_WIDTH
        self._pad_x2 = WIDTH  // 2 + PAD_HALF_WIDTH

        self._screen = None
        self._clock  = None
        self._font   = None
        if self.render_mode == "human":
            self._init_pygame()

    # ── public helper called by main.py ───────────────────────────────
    def pin_pad(self, centre_x):
        """
        Lock the landing pad to centre_x for all future resets.
        Pass None to restore random placement.
        """
        half = PAD_HALF_WIDTH
        if centre_x is None:
            self._pinned_pad_x = None
        else:
            cx = int(_clip(centre_x, half + 20, WIDTH - half - 20))
            self._pinned_pad_x = cx
            self._pad_x1 = cx - half
            self._pad_x2 = cx + half

    # ──────────────────────────────────────────
    # GYM API
    # ──────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # ── Place the landing pad ──────────────────────────────────────
        half = PAD_HALF_WIDTH
        if self._pinned_pad_x is not None:
            cx = self._pinned_pad_x
        elif self.randomize_pad:
            cx = random.randint(half + 60, WIDTH - half - 60)
        else:
            cx = WIDTH // 2

        self._pad_x1 = cx - half
        self._pad_x2 = cx + half

        pad_cx  = (self._pad_x1 + self._pad_x2) / 2
        start_x = pad_cx + random.uniform(-250, 250)
        start_x = _clip(start_x, 80, WIDTH - 80)
        start_y = random.uniform(80, 200)

        self._state = dict(
            lander_x = float(start_x),
            lander_y = float(start_y),
            vx       = random.uniform(-0.5, 0.5),
            vy       = random.uniform(0.0, 1.0),
            angle    = random.uniform(-20.0, 20.0),
            fuel     = MAX_FUEL,
            status   = "FLYING",
        )
        self._steps    = 0
        self._fuel_used = 0.0   # FIX 4: reset fuel tracking each episode
        self._prev_dist_to_pad = None   # reset approach shaping

        return self._get_obs(), self._get_info()

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action {action}"

        s = self._state

        thrusting = False
        fuel_used = 0.0

        if action == 1 and s["fuel"] > 0:
            rad        = math.radians(s["angle"])
            s["vx"]   -= math.sin(rad) * THRUST_POWER
            s["vy"]   -= math.cos(rad) * THRUST_POWER
            s["fuel"]  = max(0.0, s["fuel"] - FUEL_COST)
            fuel_used  = FUEL_COST
            thrusting  = True
        elif action == 2:
            s["vx"] -= LATERAL_POWER
        elif action == 3:
            s["vx"] += LATERAL_POWER
        elif action == 4:
            s["angle"] += ROTATE_SPEED
        elif action == 5:
            s["angle"] -= ROTATE_SPEED

        if self.wind_enabled:
            s["vx"] += random.uniform(-0.02, 0.02)

        s["vy"]       += GRAVITY
        s["lander_x"] += s["vx"]
        s["lander_y"] += s["vy"]
        s["lander_x"]  = _clip(s["lander_x"], 60, WIDTH - 60)

        # FIX 4: accumulate fuel used this episode
        self._fuel_used += fuel_used

        terminated = False
        reward     = 0.0

        pad_cx      = (self._pad_x1 + self._pad_x2) / 2
        dist_to_pad = abs(s["lander_x"] - pad_cx)

        if s["lander_y"] >= GROUND_Y - 55:
            s["lander_y"] = GROUND_Y - 55
            on_pad  = self._pad_x1 < s["lander_x"] < self._pad_x2
            slow_vy = abs(s["vy"]) < 1.5
            upright = abs(s["angle"]) < 15

            if on_pad and slow_vy and upright:
                s["status"] = "LANDED"
                # Base landing reward + fuel conservation bonus
                reward = RW_LAND + s["fuel"] * RW_FUEL_BONUS
                # Extra bonus for an especially soft landing
                if abs(s["vy"]) < 0.5:
                    reward += RW_SOFT_LAND
                # Extra bonus for landing nearly upright
                if abs(s["angle"]) < 5:
                    reward += RW_UPRIGHT
            else:
                s["status"] = "CRASHED"
                # Partial credit: don't crash-penalise as hard if almost on pad
                pad_miss_penalty = 1.0 if not on_pad else 0.3
                reward = RW_CRASH * pad_miss_penalty

            terminated = True
            s["vy"]    = 0.0

        if not terminated:
            near_ground = max(0, 1 - (GROUND_Y - 55 - s["lander_y"]) / 300)

            # ── Time penalty (very small — don't rush the agent) ──────
            reward += RW_STEP

            # ── Fuel cost per thrust action ───────────────────────────
            reward += RW_FUEL_STEP * fuel_used

            # ── Speed penalties scale with how close to ground we are ─
            reward += RW_VY_SCALE   * abs(s["vy"])            * near_ground
            reward += RW_VX_SCALE   * abs(s["vx"])            * near_ground
            reward += RW_ANGLE_SCALE * abs(s["angle"]) / 90.0 * near_ground

            # ── Potential-based pad approach reward ───────────────────
            # Positive reward for getting closer; negative for drifting away.
            # Using potential shaping: F(s',s) = γ·Φ(s') - Φ(s)
            # where Φ(s) = -dist_to_pad (higher = closer = better)
            if self._prev_dist_to_pad is not None:
                approach_delta = self._prev_dist_to_pad - dist_to_pad
                reward += RW_PAD_APPROACH * approach_delta / WIDTH

            self._prev_dist_to_pad = dist_to_pad

            # ── Weak altitude-independent alignment nudge ─────────────
            # Always reward being above the pad horizontally
            reward -= (dist_to_pad / WIDTH) * 1.5

        self._steps += 1
        truncated = (self._steps >= MAX_STEPS)

        obs  = self._get_obs()
        info = self._get_info()
        info["fuel_used"] = fuel_used

        if self.render_mode == "human":
            self._render_frame()

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self._render_frame()

    def close(self):
        if self._screen is not None:
            pygame.quit()
            self._screen = None

    # ──────────────────────────────────────────
    # INTERNAL HELPERS
    # ──────────────────────────────────────────

    def _get_obs(self):
        s      = self._state
        pad_cx = (self._pad_x1 + self._pad_x2) / 2

        return np.array([
            _clip((s["lander_x"] - pad_cx) / (WIDTH / 2),  -1, 1),
            (s["lander_y"] - HEIGHT / 2)    / (HEIGHT / 2),
            _clip(s["vx"]    / 5,   -1, 1),
            _clip(s["vy"]    / 5,   -1, 1),
            _clip(s["angle"] / 90,  -1, 1),
            s["fuel"] / MAX_FUEL,
            _clip((s["lander_x"] - pad_cx) / PAD_HALF_WIDTH, -1, 1),
            _clip(1 - (GROUND_Y - 55 - s["lander_y"]) / (HEIGHT / 2), 0, 1),
        ], dtype=np.float32)

    def _get_info(self):
        s     = self._state

        # FIX 2: raw speed magnitude in sim units/step — no fake scaling
        speed = math.sqrt(s["vx"] ** 2 + s["vy"] ** 2)

        # FIX 4: steps survived per unit of fuel burned (higher = more efficient)
        efficiency = round(self._steps / max(self._fuel_used, 0.1), 2)

        return {
            "lander_x"  : s["lander_x"],
            "lander_y"  : s["lander_y"],
            "vx"        : s["vx"],
            "vy"        : s["vy"],
            # FIX 2: honest raw speed, always returned (not zeroed on landing)
            "speed"     : round(speed, 3),
            # FIX 1: always return the real angle — never hard-code 0 on landing
            "angle"     : round(s["angle"], 1),
            "fuel"      : s["fuel"],
            # FIX 4: fuel efficiency metric
            "efficiency": efficiency,
            "status"    : s["status"],
            "steps"     : self._steps,
            "pad_x1"    : self._pad_x1,
            "pad_x2"    : self._pad_x2,
        }

    # ──────────────────────────────────────────
    # PYGAME RENDERING (human mode only)
    # ──────────────────────────────────────────

    def _init_pygame(self):
        pygame.init()
        self._screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Lunar Lander — RL Agent")
        self._clock  = pygame.time.Clock()
        self._font   = pygame.font.SysFont("Consolas", 18)

    def _render_frame(self):
        if self._screen is None:
            self._init_pygame()

        s      = self._state
        screen = self._screen

        screen.fill((5, 5, 20))

        for i in range(150):
            rng = random.Random(i)
            x, y = rng.randint(0, WIDTH), rng.randint(0, HEIGHT // 2)
            pygame.draw.circle(screen, (180, 180, 180), (x, y), 1)

        pygame.draw.rect(screen, (60, 60, 70), (0, GROUND_Y, WIDTH, HEIGHT - GROUND_Y))
        pygame.draw.rect(screen, (0, 200, 0),
                         (self._pad_x1, GROUND_Y - 4, PAD_HALF_WIDTH * 2, 6))
        screen.blit(
            self._font.render("LANDING ZONE", True, (0, 255, 0)),
            ((self._pad_x1 + self._pad_x2) // 2 - 70, GROUND_Y + 10)
        )

        lander_rect = pygame.Rect(0, 0, 40, 40)
        lander_rect.center = (int(s["lander_x"]), int(s["lander_y"]))
        lander_surf = pygame.Surface((40, 40), pygame.SRCALPHA)
        pygame.draw.rect(lander_surf, (200, 200, 220), (0, 0, 40, 40), border_radius=6)
        rotated = pygame.transform.rotate(lander_surf, s["angle"])
        screen.blit(rotated, rotated.get_rect(center=lander_rect.center))

        hud_lines = [
            f"STATUS     : {s['status']}",
            f"FUEL       : {s['fuel']:.1f}",
            f"VY         : {s['vy']:.3f}",
            f"VX         : {s['vx']:.3f}",
            # FIX 1: real angle shown always
            f"ANGLE      : {s['angle']:.1f}°",
            f"STEPS      : {self._steps}",
            # FIX 4: show efficiency
            f"EFFICIENCY : {self._fuel_used and round(self._steps / self._fuel_used, 2):.2f} s/fuel",
            f"PAD        : X={(self._pad_x1 + self._pad_x2) // 2}  "
            f"({'PINNED' if self._pinned_pad_x else 'RANDOM'})",
        ]
        for i, line in enumerate(hud_lines):
            screen.blit(self._font.render(line, True, (0, 255, 100)), (20, 20 + i * 24))

        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()


# ──────────────────────────────────────────────
# SANITY CHECK
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Running sanity check (10 random episodes, headless)...")
    env = LunarLanderEnv(render_mode=None, randomize_pad=True, wind=False)

    for ep in range(10):
        obs, info = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"  Episode {ep+1:02d} | Status: {info['status']:<8} "
              f"| Steps: {info['steps']:4d} | Reward: {total_reward:8.2f} "
              f"| Fuel left: {info['fuel']:.1f} "
              f"| Speed: {info['speed']:.3f} u/s "
              f"| Angle: {info['angle']:.1f}° "
              f"| Efficiency: {info['efficiency']:.2f} s/fuel "
              f"| Pad X: {(info['pad_x1']+info['pad_x2'])//2}")

    env.close()
    print("\nDone.")