"""
rover_ppo_controller.py  —  PPO-based autonomous rover navigation
==================================================================
Wraps the PPOAgent to drive the LunarRover on the lunar surface.

Observation vector (8 features, matches ppo_model.pth obs_dim=8):
    0  norm_x          – rover world-X normalized to [0, 1] over world_width
    1  norm_battery     – battery percentage / 100
    2  norm_speed       – current speed / max_speed (signed)
    3  norm_heading     – heading: +1.0=East(90°), -1.0=West(270°)
    4  dist_to_rock     – distance to nearest ahead rock / max_view (0=none)
    5  dist_to_crater   – distance to nearest ahead crater / max_view (0=none)
    6  hazard_score     – local hazard index [0,1] from env
    7  boundary_margin  – how close to world end (1 = far, 0 = at boundary)

Action space (6 discrete, matches act_dim=6):
    0  IDLE   – no thrust, apply drag
    1  FWD    – full forward (East)
    2  REV    – reverse / back
    3  FWD_L  – forward + slight left turn
    4  FWD_R  – forward + slight right turn
    5  SLOW   – half-throttle forward (approach / scanning)
"""

import os
import math
import numpy as np

# ── PPO import ──────────────────────────────────────────────────────────
try:
    import sys
    _parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _parent not in sys.path:
        sys.path.insert(0, _parent)
    from ppo_agent import PPOAgent
    _PPO_AVAILABLE = True
except ImportError:
    _PPO_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════
ROVER_OBS_DIM   = 8
ROVER_ACT_DIM   = 6
MAX_SPEED       = 2.5
MAX_VIEW_DIST   = 300.0   # metres ahead for proximity sensing
ROVER_MODEL_PATH = "rover_ppo_model.pth"


# ══════════════════════════════════════════════════════════════════════
# OBSERVATION BUILDER
# ══════════════════════════════════════════════════════════════════════
def build_rover_obs(rover, env) -> np.ndarray:
    """
    Construct the 8-dimensional observation vector for the rover PPO agent.
    All values are normalised to approximately [-1, 1] or [0, 1].
    """
    world_width = getattr(env, "world_width", 3000.0)

    # 0 – normalised world position
    norm_x = rover.x / world_width

    # 1 – battery level
    norm_battery = rover.battery / 100.0

    # 2 – speed (signed, normalised)
    norm_speed = rover.speed / MAX_SPEED

    # 3 – heading encoded as +1 (East) / -1 (West)
    if abs(rover.heading - 90.0) < 1.0:
        norm_heading = 1.0
    elif abs(rover.heading - 270.0) < 1.0:
        norm_heading = -1.0
    else:
        # Continuous encoding for intermediate headings
        norm_heading = math.sin(math.radians(rover.heading))

    # 4 – distance to nearest ahead rock (normalised; 0 = none in view)
    min_rock_dist = MAX_VIEW_DIST
    facing_left = (rover.heading > 180.0)
    for rock in env.rocks:
        dx = (rover.x - rock["x"]) if facing_left else (rock["x"] - rover.x)
        if 0 < dx < MAX_VIEW_DIST:
            d = dx - rock.get("r", 20.0)
            if d < min_rock_dist:
                min_rock_dist = max(0.0, d)
    dist_to_rock = 1.0 - (min_rock_dist / MAX_VIEW_DIST)  # 1=very close, 0=far

    # 5 – distance to nearest ahead crater (normalised)
    min_crater_dist = MAX_VIEW_DIST
    for crater in env.craters:
        dx = (rover.x - crater["x"]) if facing_left else (crater["x"] - rover.x)
        if 0 < dx < MAX_VIEW_DIST:
            d = dx - crater.get("r", 30.0)
            if d < min_crater_dist:
                min_crater_dist = max(0.0, d)
    dist_to_crater = 1.0 - (min_crater_dist / MAX_VIEW_DIST)

    # 6 – hazard score from env
    hazard_score = env.get_hazard_score(rover)

    # 7 – boundary margin (how far from world end, normalised)
    boundary_margin = min(1.0, (world_width - rover.x) / 200.0)

    obs = np.array([
        norm_x,
        norm_battery,
        norm_speed,
        norm_heading,
        dist_to_rock,
        dist_to_crater,
        hazard_score,
        boundary_margin,
    ], dtype=np.float32)

    return np.clip(obs, -1.0, 1.0)


# ══════════════════════════════════════════════════════════════════════
# ACTION → CONTROLS MAPPING
# ══════════════════════════════════════════════════════════════════════
def action_to_controls(action: int):
    """
    Map discrete PPO action (0-5) to (forward_backward, turn_left_right)
    control signals fed into rover.move().

    Returns:
        (forward_backward: float, turn_left_right: float)
    """
    if action == 0:   # IDLE
        return 0.0, 0.0
    elif action == 1: # FULL FORWARD (East)
        return 1.0, 0.0
    elif action == 2: # REVERSE
        return -0.7, 0.0
    elif action == 3: # FORWARD + SLIGHT LEFT TURN
        return 0.8, -0.4
    elif action == 4: # FORWARD + SLIGHT RIGHT TURN
        return 0.8, 0.4
    elif action == 5: # SLOW FORWARD (scanning approach)
        return 0.35, 0.0
    return 0.0, 0.0


# ══════════════════════════════════════════════════════════════════════
# ROVER PPO CONTROLLER
# ══════════════════════════════════════════════════════════════════════
class RoverPPOController:
    """
    Autonomous rover navigator using PPO policy inference.

    Falls back to the rule-based RoverAutopilot if no PPO model is available.
    The PPO agent uses the same ActorCritic architecture as the lander agent
    (obs_dim=8, act_dim=6) so the ppo_model.pth architecture is compatible,
    although a dedicated rover_ppo_model.pth should be trained for best results.
    """

    # ── Avoidance FSM state ────────────────────────────────────────────
    # States: "FORWARD"  – PPO drives normally
    #         "DODGE"    – steering diagonally around an obstacle
    #         "RETURN"   – steering back to Y=1500 centerline after passing
    _avd_state      = "FORWARD"
    _avd_obs_x      = 0.0    # X centre of obstacle being dodged
    _avd_dodge_head = 90.0   # heading to use while dodging (60° or 120°)
    _avd_target_y   = 1500.0 # Y to return to after dodge

    def __init__(self, model_path: str = ROVER_MODEL_PATH):
        self.ppo_agent    = None
        self.model_loaded = False
        self._fallback    = None

        # Avoidance FSM instance state
        self._avd_state      = "FORWARD"
        self._avd_obs_x      = 0.0
        self._avd_dodge_head = 90.0
        self._avd_target_y   = 1500.0

        self._try_load(model_path)

    # ──────────────────────────────────────────────────────────────────
    def _try_load(self, path: str):
        """Attempt to load a rover PPO model; silently fall back if not found."""
        if not _PPO_AVAILABLE:
            print("[RoverPPO] ppo_agent module not found – using rule-based fallback.")
            return

        if not os.path.exists(path):
            # Try loading the lander model as a warm-start for compatible architecture
            fallback_path = "ppo_model.pth"
            if os.path.exists(fallback_path):
                try:
                    import torch
                    import io, contextlib
                    ck = torch.load(fallback_path, map_location="cpu", weights_only=False)
                    hp = ck.get("hp", {})
                    if hp.get("obs_dim", 0) == ROVER_OBS_DIM and hp.get("act_dim", 0) == ROVER_ACT_DIM:
                        # Suppress any non-ASCII print from PPOAgent.load()
                        with contextlib.redirect_stdout(io.StringIO()):
                            self.ppo_agent = PPOAgent.load(fallback_path)
                        self.ppo_agent.eval()
                        self.model_loaded = True
                        print(f"[RoverPPO] Loaded lander model as rover base: {fallback_path}")
                        return
                except Exception as e:
                    print(f"[RoverPPO] Could not use lander model for rover: {type(e).__name__}")

            # No compatible model found → create fresh agent (random policy)
            print("[RoverPPO] No rover model found – initialising fresh PPO agent (untrained).")
            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                self.ppo_agent = PPOAgent(
                    obs_dim=ROVER_OBS_DIM,
                    act_dim=ROVER_ACT_DIM,
                    device="auto",
                )
            self.ppo_agent.eval()
            self.model_loaded = False
            print("[RoverPPO] Fresh PPO agent ready (rule-based collision safety active).")
            return

        try:
            self.ppo_agent = PPOAgent.load(path)
            self.ppo_agent.eval()
            self.model_loaded = True
            print(f"[RoverPPO] Rover PPO model loaded ← {path}")
        except Exception as e:
            print(f"[RoverPPO] Failed to load {path}: {e} – using rule-based fallback.")

    # ──────────────────────────────────────────────────────────────────
    def _get_fallback(self):
        """Lazy-load the rule-based autopilot as a fallback."""
        if self._fallback is None:
            from rover.rover_navigation import RoverAutopilot
            self._fallback = RoverAutopilot()
        return self._fallback

    # ──────────────────────────────────────────────────────────────────
    def compute_controls(self, rover, env, dt: float):
        """
        Compute (forward_backward, turn_left_right) control signals.

        Obstacle avoidance uses a 3-state FSM — the rover NEVER reverses:

          FORWARD  – PPO drives at full speed on a clear path (heading=90°)
          DODGE    – obstacle detected ahead; rover steers diagonally at ±30°
                     from East to pass beside it (full forward throttle)
          RETURN   – obstacle is behind us; steer back to Y=1500 centreline

        Because check_collisions() is purely a scanning mechanic (no physics
        blocking), the rover can cross any obstacle proximity zone safely.
        The 30° dodge heading gives ~175 px of lateral displacement over the
        typical 300 px X travel needed to clear a rock or crater.
        """
        if rover.battery <= 0.0:
            return 0.0, 0.0

        # ── Rule-based fallback ──────────────────────────────────────
        if self.ppo_agent is None:
            return self._get_fallback().compute_controls(rover, env, dt)

        ROVER_HALF  = 55.0    # half rover sprite width (px)
        DETECT_DIST = 180.0   # begin dodge when obstacle centre is this far ahead
        DODGE_DEG   = 30.0    # degrees off East for dodge heading
        CENTER_Y    = 1500.0  # nominal path Y in world coords
        CENTER_TOL  = 8.0     # snap back to centre within this Y margin
        world_width = getattr(env, "world_width", 3000.0)

        # ── Boundary guard ───────────────────────────────────────────
        if rover.x > world_width - 80.0:
            rover.status  = "SURFACE SURVEY COMPLETE"
            rover.heading = 90.0
            return 0.0, 0.0

        # ────────────────────────────────────────────────────────────
        # FSM: RETURN — steer back to centreline after passing obstacle
        # ────────────────────────────────────────────────────────────
        if self._avd_state == "RETURN":
            dy = CENTER_Y - rover.y
            if abs(dy) <= CENTER_TOL:
                rover.heading   = 90.0
                self._avd_state = "FORWARD"
            else:
                # dy > 0 → rover is above centre (low y) → steer down (heading > 90°)
                # dy < 0 → rover is below centre (high y) → steer up (heading < 90°)
                rover.heading = 90.0 + (DODGE_DEG * 0.6 if dy > 0 else -DODGE_DEG * 0.6)
            rover.status = "RETURNING TO PATH [PPO]"
            return 1.0, 0.0

        # ────────────────────────────────────────────────────────────
        # FSM: DODGE — steer diagonally past the flagged obstacle
        # ────────────────────────────────────────────────────────────
        if self._avd_state == "DODGE":
            # Switch to RETURN once rover X has passed the obstacle centre + margin
            if rover.x > self._avd_obs_x + 80.0:
                self._avd_state = "RETURN"
                rover.status    = "RETURNING TO PATH [PPO]"
                return 1.0, 0.0
            rover.heading = self._avd_dodge_head
            rover.status  = "DODGING OBSTACLE [PPO]"
            return 1.0, 0.0

        # ────────────────────────────────────────────────────────────
        # FSM: FORWARD — scan ahead, let PPO drive when path is clear
        # ────────────────────────────────────────────────────────────
        nearest_obs = None
        nearest_dx  = float("inf")

        for obs in env.rocks + env.craters:
            dx_centre = obs["x"] - rover.x
            if dx_centre <= 0:
                continue                        # obstacle is behind rover
            obs_r = obs.get("r", 25.0)

            # 2-D: only dodge if obstacle is actually in the rover's current Y track
            dy = abs(obs["y"] - rover.y)
            if dy >= ROVER_HALF + obs_r:
                continue                        # rover passes beside it

            if dx_centre < nearest_dx:
                nearest_dx  = dx_centre
                nearest_obs = obs

        if nearest_obs is not None and nearest_dx < DETECT_DIST:
            # Choose dodge direction away from the obstacle's Y position
            obs_y = nearest_obs["y"]
            if obs_y <= CENTER_Y:
                # Obstacle at or above centreline → dodge downward (heading > 90°)
                self._avd_dodge_head = 90.0 + DODGE_DEG
            else:
                # Obstacle below centreline → dodge upward (heading < 90°)
                self._avd_dodge_head = 90.0 - DODGE_DEG

            self._avd_state = "DODGE"
            self._avd_obs_x = nearest_obs["x"]
            rover.heading   = self._avd_dodge_head
            rover.status    = "DODGING OBSTACLE [PPO]"
            return 1.0, 0.0

        # ── Clear path — PPO drives ──────────────────────────────────
        rover.heading = 90.0
        obs_vec = build_rover_obs(rover, env)
        action  = self.ppo_agent.select_action(obs_vec)
        fwd, _  = action_to_controls(action)   # ignore PPO turn; we manage heading

        # Always move forward (suppress random PPO idling/reversals)
        fwd = max(0.7, fwd) if fwd > 0 else 0.8

        rover.status = "EXPLORING [PPO]"
        return fwd, 0.0

    # ──────────────────────────────────────────────────────────────────
    def save_model(self, path: str = ROVER_MODEL_PATH):
        """Save the current PPO model weights to disk."""
        if self.ppo_agent is not None:
            self.ppo_agent.save(path)

    @property
    def is_ppo_active(self) -> bool:
        """True when PPO is driving, False when rule-based fallback is used."""
        return self.ppo_agent is not None
