"""
main.py  —  Lunar Lander Renderer
===================================
Pure rendering / game-loop module.
Does NOT contain training logic.

Modes (press M to cycle):
  MANUAL  →  keyboard control
  AI      →  built-in rule-based autopilot
  PPO     →  loaded PPO agent (requires ppo_agent.py + saved model)

Run directly:
    python main.py                  # starts in MANUAL mode
    python main.py --mode ppo       # starts in PPO mode (needs model)
    python main.py --mode ai        # starts in AI mode

FIXES APPLIED:
    - Bug 2: HUD label changed from "km/h" to "u/s" to match the real raw speed value
    - Bug 3: update_history() now receives ep_reward (cumulative) not per-step reward,
             so the telemetry graph reward line is meaningful
    - Bug 4: EFFIC field added to draw_hud() displaying fuel efficiency from env info
"""

import sys
import math
import random
import argparse
import numpy as np
import pygame

from lunar_lander_env import LunarLanderEnv, GROUND_Y, WIDTH, HEIGHT, PAD_HALF_WIDTH

# ── Mission and Rover Imports ──────────────────────────────────────────
from mission.mission_controller import MissionController, MissionState
from rover.rover import LunarRover
from rover.rover_env import RoverEnv
from rover.rover_navigation import RoverAutopilot
from rover.rover_ppo_controller import RoverPPOController
from rover.rover_telemetry import RoverTelemetryLogger

# ── Optional PPO import ────────────────────────────────────────────────
try:
    from ppo_agent import PPOAgent
    PPO_AVAILABLE = True
except ImportError:
    PPO_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description="Lunar Lander Renderer")
parser.add_argument("--mode",  default="manual", choices=["manual", "ai", "ppo"])
parser.add_argument("--model", default="ppo_model.pth")
parser.add_argument("--phase", default="landing", choices=["landing", "rover"])
parser.add_argument("--test-frames", type=int, default=0)
parser.add_argument("--save-screenshot", default="")
args = parser.parse_args()


# ══════════════════════════════════════════════════════════════════════
# PYGAME INIT
# ══════════════════════════════════════════════════════════════════════
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Lunar Lander  |  RL Demo")
clock = pygame.time.Clock()


# ══════════════════════════════════════════════════════════════════════
# ASSETS
# ══════════════════════════════════════════════════════════════════════
def load_asset(path, size, alpha=False):
    try:
        img = pygame.image.load(path)
        img = img.convert_alpha() if alpha else img.convert()
        return pygame.transform.smoothscale(img, size)
    except FileNotFoundError:
        surf = pygame.Surface(size, pygame.SRCALPHA if alpha else 0)
        surf.fill((80, 80, 100))
        return surf

moon_surf  = load_asset("assets/moon_surface.jpg", (WIDTH, 300))
lander_img = load_asset("assets/lander.png",       (110, 110), alpha=True)
earth_img  = load_asset("assets/earth.png",        (120, 120), alpha=True)
rover_img  = load_asset("assets/rover.png",        (105, 70), alpha=True)


# ══════════════════════════════════════════════════════════════════════
# FONTS
# ══════════════════════════════════════════════════════════════════════
font      = pygame.font.SysFont("Consolas", 20)
font_sm   = pygame.font.SysFont("Consolas", 14)
font_big  = pygame.font.SysFont("Consolas", 36)
font_warn = pygame.font.SysFont("Consolas", 28, bold=True)


# ══════════════════════════════════════════════════════════════════════
# AUDIO
# ══════════════════════════════════════════════════════════════════════
pygame.mixer.init()
warning_alarm_playing = False

try:
    warning_alarm = pygame.mixer.Sound("assets/warning.mp3")
    warning_alarm.set_volume(1.0)
except Exception as e:
    print("Audio load error:", e)
    warning_alarm = None


def stop_alarm():
    """Unconditionally stop the warning alarm."""
    global warning_alarm_playing
    if warning_alarm and warning_alarm_playing:
        warning_alarm.stop()
        warning_alarm_playing = False


# ══════════════════════════════════════════════════════════════════════
# STARS
# ══════════════════════════════════════════════════════════════════════
stars = [
    (random.randint(0, WIDTH), random.randint(0, HEIGHT),
     random.choice([1, 2]), random.randint(150, 255))
    for _ in range(200)
]


# ══════════════════════════════════════════════════════════════════════
# PARTICLES
# ══════════════════════════════════════════════════════════════════════
dust_particles      = []
explosion_particles = []

def spawn_dust(x, y, vx, vy):
    speed = abs(vx) + abs(vy)
    for _ in range(int(4 + speed * 3)):
        dust_particles.append([
            x + random.randint(-30, 30), y,
            random.uniform(-1.5, 1.5) + vx * 0.3,
            random.uniform(-3, -1) - abs(vy) * 0.3,
            random.randint(3, 6)
        ])

def update_dust():
    for p in dust_particles[:]:
        p[0] += p[2]; p[1] += p[3]; p[3] += 0.15; p[4] -= 0.08
        if p[4] <= 0:
            dust_particles.remove(p)

def draw_dust():
    for x, y, _, _, size in dust_particles:
        pygame.draw.circle(screen, (160, 160, 160), (int(x), int(y)), int(size))

def spawn_explosion(x, y, vx, vy):
    for _ in range(80):
        explosion_particles.append([
            x, y,
            random.uniform(-4, 4) + vx * 0.5,
            random.uniform(-4, 1) + vy * 0.5,
            random.randint(4, 8),
            random.choice([(255, 80, 0), (255, 140, 0), (255, 200, 0)])
        ])

def update_explosion():
    for p in explosion_particles[:]:
        p[0] += p[2]; p[1] += p[3]; p[3] += 0.15; p[4] -= 0.15
        if p[4] <= 0:
            explosion_particles.remove(p)

def draw_explosion():
    for x, y, _, _, size, color in explosion_particles:
        pygame.draw.circle(screen, color, (int(x), int(y)), int(size))


# ══════════════════════════════════════════════════════════════════════
# FLAME RENDERING
# ══════════════════════════════════════════════════════════════════════
NOZZLE_LOCAL = (-6, 36)

def world_nozzle(cx, cy, angle_deg, local=NOZZLE_LOCAL):
    rad = math.radians(-angle_deg)
    lx, ly = local
    wx = cx + lx * math.cos(rad) - ly * math.sin(rad)
    wy = cy + lx * math.sin(rad) + ly * math.cos(rad)
    return wx, wy

def make_flame_surface(length=None, wide=14):
    length = length or random.randint(30, 45)
    surf = pygame.Surface((wide * 6, length), pygame.SRCALPHA)
    for i in range(length):
        t = i / length
        w = int(wide * (1 - t * 0.9))
        if   t < 0.2: color = (255, 255, 255, 255)
        elif t < 0.5: color = (255, 220, 120, 230)
        elif t < 0.8: color = (255, 140,   0, 200)
        else:         color = (255,  60,   0, 120)
        pygame.draw.ellipse(surf, color,
                            (surf.get_width() // 2 - w // 2, i, w, 4))
    return surf

def draw_main_flame(cx, cy, angle_deg):
    surf    = make_flame_surface()
    rotated = pygame.transform.rotate(surf, angle_deg)
    nx, ny  = world_nozzle(cx, cy, angle_deg)
    rad     = math.radians(-angle_deg)
    off_x   = math.sin(rad) * surf.get_height() / 2
    off_y   = math.cos(rad) * surf.get_height() / 2
    screen.blit(rotated, rotated.get_rect(center=(nx + off_x, ny + off_y)))

def draw_side_flame(cx, cy, angle_deg, side):
    surf = make_flame_surface(length=20, wide=8)
    if side == "left":
        base_angle, local_offset = 90,  (35, 0)
    else:
        base_angle, local_offset = -90, (-35, 0)
    rotated = pygame.transform.rotate(surf, base_angle - angle_deg)
    wx, wy  = world_nozzle(cx, cy, angle_deg, local=local_offset)
    screen.blit(rotated, rotated.get_rect(center=(wx, wy)))


# ══════════════════════════════════════════════════════════════════════
# SCENE DRAW HELPERS
# ══════════════════════════════════════════════════════════════════════
def draw_stars():
    for x, y, size, b in stars:
        pygame.draw.circle(screen, (b, b, b), (x, y), size)

def draw_earth():
    screen.blit(earth_img, (WIDTH - 200, 40))

def draw_surface():
    screen.blit(moon_surf, (0, GROUND_Y))
    overlay = pygame.Surface((WIDTH, 300), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 120))
    screen.blit(overlay, (0, GROUND_Y))

def draw_landing_zone(pad_x1, pad_x2):
    for x in range(pad_x1, pad_x2, 12):
        pygame.draw.line(screen, (0, 255, 0),
                         (x, GROUND_Y - 2), (x + 6, GROUND_Y - 2), 2)

    post_height = 45
    for px in (pad_x1, pad_x2):
        pygame.draw.line(screen, (220, 220, 220),
                         (px, GROUND_Y - post_height), (px, GROUND_Y), 3)

    # left flag
    pygame.draw.polygon(screen, (255, 60, 60), [
        (pad_x1, GROUND_Y - post_height),
        (pad_x1 + 20, GROUND_Y - post_height + 8),
        (pad_x1, GROUND_Y - post_height + 16),
    ])
    # right flag
    pygame.draw.polygon(screen, (255, 60, 60), [
        (pad_x2, GROUND_Y - post_height),
        (pad_x2 - 20, GROUND_Y - post_height + 8),
        (pad_x2, GROUND_Y - post_height + 16),
    ])

    cx = (pad_x1 + pad_x2) // 2
    label = font.render(f"LANDING ZONE  (X={cx})", True, (0, 255, 0))
    screen.blit(label, (cx - label.get_width() // 2, GROUND_Y + 15))

def draw_lander(x, y, angle):
    rotated = pygame.transform.rotate(lander_img, angle)
    screen.blit(rotated, rotated.get_rect(center=(int(x), int(y))))


# ══════════════════════════════════════════════════════════════════════
# WARNING SYSTEM
# ══════════════════════════════════════════════════════════════════════
def get_guidance_warnings(info):
    warnings = []
    altitude = GROUND_Y - 55 - info.get("lander_y", 0)
    vx    = abs(info.get("vx", 0))
    vy    = info.get("vy", 0)
    angle = abs(info.get("angle", 0))
    fuel  = info.get("fuel", 100)
    speed = math.sqrt(info.get("vx", 0)**2 + info.get("vy", 0)**2)

    if fuel < 20:
        warnings.append(("LOW FUEL", "warning"))
    if fuel < 8:
        warnings.append(("CRITICAL FUEL", "critical"))
    if vy > 2.2:
        warnings.append(("HARD DESCENT", "critical"))
    if speed > 3.0:
        warnings.append(("OVERSPEED", "critical"))
    if vx > 1.2:
        warnings.append(("HIGH LATERAL VELOCITY", "warning"))
    if angle > 20:
        warnings.append(("CRITICAL ATTITUDE", "critical"))
    if altitude < 100 and vy > 1.8:
        warnings.append(("IMPACT IMMINENT", "critical"))
    return warnings


def draw_warning_system(warnings):
    global warning_alarm_playing

    if not warnings:
        stop_alarm()
        return

    blink           = (pygame.time.get_ticks() // 400) % 2 == 0
    critical_exists = any(level == "critical" for _, level in warnings)

    if critical_exists:
        if warning_alarm and not warning_alarm_playing:
            warning_alarm.play(-1)
            warning_alarm_playing = True
    else:
        stop_alarm()

    if critical_exists and blink:
        banner = pygame.Surface((WIDTH, 50), pygame.SRCALPHA)
        banner.fill((255, 0, 0, 180))
        screen.blit(banner, (0, 0))
        txt = font_warn.render("CRITICAL FLIGHT WARNING", True, (255, 255, 255))
        screen.blit(txt, ((WIDTH - txt.get_width()) // 2, 10))

    panel = pygame.Surface((320, 220), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 190))
    screen.blit(panel, (WIDTH - 350, 340))
    pygame.draw.rect(screen, (255, 80, 80), (WIDTH - 350, 340, 320, 220), 2)
    title = font.render("MISSION WARNINGS", True, (255, 120, 120))
    screen.blit(title, (WIDTH - 330, 350))

    y = 385
    for msg, level in warnings:
        color = ((255, 0, 0) if blink else (255, 255, 255)) if level == "critical" else (255, 180, 0)
        screen.blit(font_sm.render(msg, True, color), (WIDTH - 330, y))
        y += 22


# ══════════════════════════════════════════════════════════════════════
# COORDINATE INPUT OVERLAY
# ══════════════════════════════════════════════════════════════════════
coord_input_active = False
coord_input_text   = ""
coord_input_error  = ""

def draw_coord_input():
    pw, ph = 460, 150
    px = WIDTH // 2 - pw // 2
    py = HEIGHT // 2 - ph // 2

    panel = pygame.Surface((pw, ph), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 230))
    screen.blit(panel, (px, py))
    pygame.draw.rect(screen, (0, 200, 255), (px, py, pw, ph), 2)

    title = font.render("SET LANDING PAD  X-COORDINATE", True, (0, 200, 255))
    screen.blit(title, (px + pw // 2 - title.get_width() // 2, py + 12))

    hint = font_sm.render(
        f"Range: {PAD_HALF_WIDTH + 20} – {WIDTH - PAD_HALF_WIDTH - 20}   "
        f"[ENTER] confirm   [ESC] cancel",
        True, (140, 140, 140)
    )
    screen.blit(hint, (px + pw // 2 - hint.get_width() // 2, py + 42))

    box = pygame.Rect(px + 80, py + 70, pw - 160, 36)
    pygame.draw.rect(screen, (15, 15, 35), box)
    pygame.draw.rect(screen, (0, 200, 255), box, 2)
    cursor = "|" if (pygame.time.get_ticks() // 500) % 2 == 0 else " "
    screen.blit(font.render(coord_input_text + cursor, True, (255, 255, 255)),
                (box.x + 10, box.y + 7))

    if coord_input_error:
        err = font_sm.render(coord_input_error, True, (255, 80, 80))
        screen.blit(err, (px + pw // 2 - err.get_width() // 2, py + 120))


# ══════════════════════════════════════════════════════════════════════
# HUD / CONTROLS / RESULT
# ══════════════════════════════════════════════════════════════════════
def draw_hud(info, mode_name, episode, ep_reward):
    lines = [
        f"MODE:    {mode_name}",
        f"EPISODE: {episode}",
        f"REWARD:  {ep_reward:.1f}",
        f"STATUS:  {info.get('status', '?')}",
        f"FUEL:    {info.get('fuel', 0):.1f}",
        f"VY:      {info.get('vy', 0):.3f}",
        f"VX:      {info.get('vx', 0):.3f}",
        # FIX 2: label now correctly says u/s (sim units per step), not km/h
        f"SPEED:   {info.get('speed', 0):.3f} u/s",
        # FIX 1: angle is always the real value from env (never fake 0)
        f"ANGLE:   {info.get('angle', 0):.1f}°",
        f"STEPS:   {info.get('steps', 0)}",
        # FIX 4: fuel efficiency displayed
        f"EFFIC:   {info.get('efficiency', 0):.2f} s/fuel",
    ]
    for i, line in enumerate(lines):
        screen.blit(font.render(line, True, (0, 255, 0)), (20, 20 + i * 25))

def draw_controls_hint(mode_name):
    hints = [
        "[↑] Thrust  [←][→] Drift  [A][D] Rotate",
        "[R] Restart  [M] Mode  [G] Graph  [C] Set Pad X",
        f"Mode: {mode_name}",
    ]
    for i, h in enumerate(hints):
        screen.blit(font_sm.render(h, True, (100, 200, 100)),
                    (WIDTH - 490, HEIGHT - 72 + i * 22))

def draw_result(status, ep_reward=0.0, fuel=0.0):
    if status == "LANDED":
        msg, color = "LANDED SAFELY!", (0, 255, 100)
    elif status == "CRASHED":
        msg, color = "CRASHED!", (255, 80, 0)
    else:
        return
    txt = font_big.render(msg, True, color)
    screen.blit(txt, (WIDTH // 2 - txt.get_width() // 2, HEIGHT // 2 - 80))
    # Show reward and fuel summary
    summary = font.render(
        f"Episode Reward: {ep_reward:.1f}   |   Fuel Left: {fuel:.1f}",
        True, (220, 220, 220)
    )
    screen.blit(summary, (WIDTH // 2 - summary.get_width() // 2, HEIGHT // 2 - 30))
    sub = font.render("Press [R] to restart", True, (200, 200, 200))
    screen.blit(sub, (WIDTH // 2 - sub.get_width() // 2, HEIGHT // 2 + 10))


# ══════════════════════════════════════════════════════════════════════
# TELEMETRY GRAPH
# ══════════════════════════════════════════════════════════════════════
show_graph   = False
history_vy   = []
history_fuel = []
history_rew  = []
MAX_HIST     = 300

def update_history(vy, fuel, reward):
    for hist, val in [(history_vy, vy), (history_fuel, fuel), (history_rew, reward)]:
        hist.append(val)
        if len(hist) > MAX_HIST:
            hist.pop(0)

def draw_graph_panel():
    pw, ph = 460, 240
    px, py = WIDTH // 2 - pw // 2, HEIGHT // 2 - ph // 2
    panel  = pygame.Surface((pw, ph), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 210))
    screen.blit(panel, (px, py))
    pygame.draw.rect(screen, (0, 255, 0), (px, py, pw, ph), 1)
    screen.blit(font.render("TELEMETRY  [G] to close", True, (0, 255, 0)), (px + 8, py + 6))

    row_h = 60

    def draw_line(data, color, y_min, y_max, row_y, label):
        screen.blit(font_sm.render(label, True, color), (px + 6, row_y + 2))
        if len(data) < 2:
            return
        span = max(y_max - y_min, 0.001)
        pts = []
        for i, v in enumerate(data):
            sx = px + 55 + int(i * (pw - 65) / MAX_HIST)
            sy = row_y + row_h - int((v - y_min) / span * (row_h - 6)) - 3
            pts.append((sx, sy))
        pygame.draw.lines(screen, color, False, pts, 1)

    draw_line(history_vy,   (255, 120, 0), -5,   5,   py + 30,  "VY")
    draw_line(history_fuel, (0, 200, 255),  0,   100, py + 105, "FUEL")
    # Auto-scale reward axis so landing bonuses (now up to ~450) are always visible
    rew_min = min(history_rew) if history_rew else -300
    rew_max = max(history_rew) if history_rew else  300
    rew_pad = max(abs(rew_max - rew_min) * 0.1, 20)
    draw_line(history_rew, (180, 0, 255), rew_min - rew_pad, rew_max + rew_pad, py + 175, "REW")


# ══════════════════════════════════════════════════════════════════════
# AI AUTOPILOT  —  PD-style controller
# ══════════════════════════════════════════════════════════════════════
def rule_based_action(info, pad_x1, pad_x2):
    """
    Action space: 0=nothing  1=main thrust  2=left-drift  3=right-drift
                  4=rot_left  5=rot_right
    """
    cx   = (pad_x1 + pad_x2) / 2.0
    lx   = info.get("lander_x", cx)
    ly   = info.get("lander_y", 0)
    vx   = info.get("vx", 0)
    vy   = info.get("vy", 0)
    ang  = info.get("angle", 0)
    fuel = info.get("fuel", 0)

    dx  = cx - lx
    alt = GROUND_Y - 55 - ly

    desired_ang = max(-20.0, min(20.0, dx * 0.10 - vx * 2.0))
    ang_err = ang - desired_ang

    if ang_err > 4:
        return 5
    if ang_err < -4:
        return 4

    future_x = lx + vx * 10
    future_dx = cx - future_x

    if future_dx > 20:
        return 3
    if future_dx < -20:
        return 2

    if alt > 350:
        target_vy = 4.0
    elif alt > 200:
        target_vy = 2.5
    elif alt > 80:
        target_vy = 1.4
    else:
        target_vy = 0.6

    if vy > target_vy and fuel > 0:
        return 1

    return 0


# ══════════════════════════════════════════════════════════════════════
# KEYBOARD → ACTION
# ══════════════════════════════════════════════════════════════════════
def keyboard_action(keys):
    if keys[pygame.K_UP]:    return 1
    if keys[pygame.K_LEFT]:  return 2
    if keys[pygame.K_RIGHT]: return 3
    if keys[pygame.K_a]:     return 4
    if keys[pygame.K_d]:     return 5
    return 0


# ══════════════════════════════════════════════════════════════════════
# PPO AGENT
# ══════════════════════════════════════════════════════════════════════
ppo_agent = None

def try_load_ppo(model_path):
    global ppo_agent
    if not PPO_AVAILABLE:
        print("[WARN] ppo_agent.py not found — PPO mode unavailable.")
        return False
    try:
        ppo_agent = PPOAgent.load(model_path)
        ppo_agent.eval()
        print(f"[INFO] PPO model loaded from {model_path}")
        return True
    except Exception as e:
        print(f"[WARN] Could not load PPO model: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════
# ROVER SCENE DRAWING
# ══════════════════════════════════════════════════════════════════════
def draw_rover_scene(dt):
    # Fill background
    screen.fill((5, 5, 15))
    
    # ── Draw Stars ──
    draw_stars()
    
    # ── Header ──
    # Top border / line
    pygame.draw.line(screen, (0, 150, 255), (50, 60), (1150, 60), 2)
    title_lbl = font_big.render("LUNAR EXPLORATION MISSION CONTROL", True, (0, 200, 255))
    screen.blit(title_lbl, (50, 15))
    
    # COMMS STATUS
    comms_color = (0, 255, 100) if (pygame.time.get_ticks() // 800) % 2 == 0 or rover.battery > 0 else (100, 100, 100)
    comms_lbl = font.render("COMMS: CONNECTED" if rover.battery > 0 else "COMMS: OFFLINE", True, comms_color)
    screen.blit(comms_lbl, (1150 - comms_lbl.get_width(), 25))
    
    # ── Live Rover View Panel ──
    # Panel box
    live_view_rect = pygame.Rect(50, 80, 1100, 400)
    pygame.draw.rect(screen, (0, 100, 180), live_view_rect, 2)
    
    # Clip drawing to viewport
    screen.set_clip(pygame.Rect(52, 82, 1096, 396))
    
    # Fill background inside viewport with space black
    screen.fill((0, 0, 10))
    
    # Render stars in viewport
    for x, y, size, b in stars:
        pygame.draw.circle(screen, (b, b, b), (x % 1096 + 52, y % 220 + 82), size)
        
    # Render Earth inside viewport
    screen.blit(pygame.transform.smoothscale(earth_img, (65, 65)), (980, 95))
    
    # Render ground surface
    surface_y = 380
    bg_w = 1096
    
    # Camera position: keep rover on the left side of the screen
    # At start (rover.x = 280), cam_x = 0, rover screen X = 280 - 0 + 52 = 332
    cam_x = max(0.0, rover.x - 220.0)
    x_offset = -int(cam_x) % bg_w
    
    screen.blit(pygame.transform.smoothscale(moon_surf, (bg_w, 120)), (52 + x_offset - bg_w, surface_y))
    screen.blit(pygame.transform.smoothscale(moon_surf, (bg_w, 120)), (52 + x_offset, surface_y))
    screen.blit(pygame.transform.smoothscale(moon_surf, (bg_w, 120)), (52 + x_offset + bg_w, surface_y))
    
    # Landing site pad & flags (fixed world position on the left)
    pad_cx = 160.0
    pad_half = 65
    pad_x1 = int(pad_cx - pad_half - cam_x) + 52
    pad_x2 = int(pad_cx + pad_half - cam_x) + 52
    for px in range(pad_x1, pad_x2, 12):
        if 52 <= px < 1148:
            pygame.draw.line(screen, (0, 255, 0), (px, surface_y), (px + 6, surface_y), 2)
            
    # Draw landing pad posts & flags
    post_height = 45
    for px in (pad_x1, pad_x2):
        if 52 <= px < 1148:
            pygame.draw.line(screen, (220, 220, 220), (px, surface_y - post_height), (px, surface_y), 3)
            if px == pad_x1:
                pygame.draw.polygon(screen, (255, 60, 60), [
                    (px, surface_y - post_height),
                    (px + 20, surface_y - post_height + 8),
                    (px, surface_y - post_height + 16),
                ])
            else:
                pygame.draw.polygon(screen, (255, 60, 60), [
                    (px, surface_y - post_height),
                    (px - 20, surface_y - post_height + 8),
                    (px, surface_y - post_height + 16),
                ])
                
    # Draw the landed lander spacecraft next to the pad center
    lander_screen_x = int(pad_cx - cam_x) + 52
    if -100 <= lander_screen_x < 1200:
        lander_w, lander_h = 90, 90
        scaled_lander = pygame.transform.smoothscale(lander_img, (lander_w, lander_h))
        screen.blit(scaled_lander, (lander_screen_x - lander_w // 2, surface_y - lander_h + 10))

    # ── Draw Rocks and Craters on the Main Lunar Surface ──
    # 1. Craters (surface depressions)
    for crater in rover_env.craters:
        cx_screen = int(crater["x"] - cam_x) + 52
        if -100 <= cx_screen <= 1250:
            c_w = int(crater["r"] * 1.5)
            c_h = int(crater["r"] * 0.40)
            c_rect = pygame.Rect(cx_screen - c_w // 2, surface_y - 2, c_w, c_h)
            pygame.draw.ellipse(screen, (15, 16, 24), c_rect)
            pygame.draw.arc(screen, (150, 155, 170), c_rect, 0.2, 3.14, 2)
            pygame.draw.arc(screen, (40, 42, 50), c_rect, 3.14, 6.28, 1)
            if crater.get("scanned", False):
                pygame.draw.circle(screen, (0, 240, 160), (cx_screen, surface_y + 4), 3)

    # 2. Rocks & Boulders (standing on the surface)
    for rock in rover_env.rocks:
        rx_screen = int(rock["x"] - cam_x) + 52
        if -60 <= rx_screen <= 1200:
            r_sz = rock["r"] * 0.95
            sh_rect = pygame.Rect(int(rx_screen - r_sz * 0.8), int(surface_y - 2), int(r_sz * 1.6), int(r_sz * 0.35))
            pygame.draw.ellipse(screen, (12, 14, 20), sh_rect)
            
            poly_pts = []
            for fx, fy in rock.get("facets", []):
                poly_pts.append((int(rx_screen + fx * r_sz * 0.85), int(surface_y - 4 + fy * r_sz * 0.75)))
            
            if len(poly_pts) >= 3:
                tint = rock.get("tint", 135)
                pygame.draw.polygon(screen, (tint, tint, tint + 5), poly_pts)
                hi_pts = [poly_pts[4], poly_pts[5], poly_pts[6], (rx_screen, int(surface_y - 10))]
                pygame.draw.polygon(screen, (min(255, tint + 55), min(255, tint + 55), min(255, tint + 60)), hi_pts)
                sh_pts = [poly_pts[0], poly_pts[1], poly_pts[2], (rx_screen, int(surface_y - 10))]
                pygame.draw.polygon(screen, (max(20, tint - 60), max(20, tint - 60), max(25, tint - 55)), sh_pts)
                pygame.draw.polygon(screen, (25, 28, 35), poly_pts, 1)

            if rock.get("scanned", False):
                pygame.draw.circle(screen, (0, 240, 160), (rx_screen, int(surface_y - r_sz * 0.8)), 3)

    # ── Draw Rover (Positioned on Left Side of Screen, Facing Heading) ──
    rover_screen_x = int(rover.x - cam_x) + 52
    rover_w, rover_h = 105, 70
    scaled_rover = pygame.transform.smoothscale(rover_img, (rover_w, rover_h))
    if rover.heading == 270.0:
        scaled_rover = pygame.transform.flip(scaled_rover, True, False)
    screen.blit(scaled_rover, (rover_screen_x - rover_w // 2, surface_y - rover_h + 8))
    
    # Reset clip
    screen.set_clip(None)
    
    # Overlay label inside view
    feed_color = (255, 80, 80) if (pygame.time.get_ticks() // 600) % 2 == 0 else (200, 50, 50)
    feed_lbl = font.render("• LIVE TELEMETRY STREAM", True, feed_color)
    screen.blit(feed_lbl, (70, 95))
    
    # ── Camera Panel PiP (Picture in Picture - Optical Hazcam) ──
    cam_pip_x, cam_pip_y = 930, 90
    cam_frame = rover_env.get_camera_frame(rover, width=200, height=120)
    screen.blit(cam_frame, (cam_pip_x, cam_pip_y))
    pygame.draw.rect(screen, (0, 200, 255), (cam_pip_x, cam_pip_y, 200, 120), 1)
    cam_lbl = font_sm.render("ROVER FWD HAZCAM // MACHINE VISION", True, (0, 200, 255))
    screen.blit(cam_lbl, (cam_pip_x + (200 - cam_lbl.get_width()) // 2, cam_pip_y + 125))

    # ── Bottom Panels (3 Columns) ──
    # Column 1: ROVER STATUS
    box1 = pygame.Rect(50, 500, 350, 170)
    pygame.draw.rect(screen, (0, 100, 180), box1, 1)
    shadow1 = pygame.Surface((348, 168))
    shadow1.fill((10, 10, 25))
    screen.blit(shadow1, (51, 501))
    
    title1 = font.render("ROVER STATUS", True, (0, 200, 255))
    screen.blit(title1, (70, 510))
    
    bat_color = (0, 255, 100) if rover.battery > 50 else ((255, 150, 0) if rover.battery > 20 else (255, 50, 50))
    pygame.draw.rect(screen, (40, 40, 50), (70, 560, 200, 10))
    pygame.draw.rect(screen, bat_color, (70, 560, int(200 * (rover.battery / 100.0)), 10))
    
    lines1 = [
        f"Battery: {rover.battery:.1f}%",
        f"Speed:   {abs(rover.speed):.2f} m/s",
        f"Temp:    {rover.temperature:.1f} C",
        f"Status:  {rover.status}"
    ]
    for idx, l in enumerate(lines1):
        offset_y = 540 if idx == 0 else (580 + (idx - 1) * 22)
        screen.blit(font_sm.render(l, True, (200, 220, 255)), (70, offset_y))
        
    # Column 2: TERRAIN ANALYSIS
    box2 = pygame.Rect(425, 500, 350, 170)
    pygame.draw.rect(screen, (0, 100, 180), box2, 1)
    shadow2 = pygame.Surface((348, 168))
    shadow2.fill((10, 10, 25))
    screen.blit(shadow2, (426, 501))
    
    title2 = font.render("TERRAIN ANALYSIS", True, (0, 200, 255))
    screen.blit(title2, (445, 510))
    
    hz_score = rover_env.get_hazard_score(rover)
    terrain_type = rover_env.get_terrain_type(rover)
    
    lines2 = [
        f"Terrain: {terrain_type}",
        f"Rocks scanned:   {rover.scanned_rocks}",
        f"Craters scanned: {rover.scanned_craters}",
        f"Hazard Index:    {hz_score:.2f}"
    ]
    for idx, l in enumerate(lines2):
        screen.blit(font_sm.render(l, True, (200, 220, 255)), (445, 545 + idx * 24))

    # Column 3: MISSION STATUS
    box3 = pygame.Rect(800, 500, 350, 170)
    pygame.draw.rect(screen, (0, 100, 180), box3, 1)
    shadow3 = pygame.Surface((348, 168))
    shadow3.fill((10, 10, 25))
    screen.blit(shadow3, (801, 501))
    
    title3 = font.render("MISSION STATUS", True, (0, 200, 255))
    screen.blit(title3, (820, 510))
    
    total_targets = len(rover_env.rocks) + len(rover_env.craters)
    scanned_targets = rover.scanned_rocks + rover.scanned_craters
    coverage = (scanned_targets / total_targets * 100.0) if total_targets > 0 else 0.0
    
    ppo_status = "PPO" if rover_ppo.is_ppo_active else "RULE-BASED"
    lines3 = [
        f"Coverage: {coverage:.1f}%",
        f"Distance: {rover.distance_travelled:.1f} m",
        f"Mode:     {rover.mode}  [{ppo_status}]",
        f"Control:  [A/D] Drive  [M] Mode",
        f"Reset:    [R] Restart Rover"
    ]
    for idx, l in enumerate(lines3):
        color = (0, 255, 100) if "Mode:" in l and "AUTONOMOUS" in l else (200, 220, 255)
        if "Control:" in l or "Reverse:" in l or "Reset:" in l:
            color = (150, 170, 200)
        screen.blit(font_sm.render(l, True, color), (820, 545 + idx * 24))


# ══════════════════════════════════════════════════════════════════════
# HELPER — full episode reset
# ══════════════════════════════════════════════════════════════════════
def do_reset(env, pinned_pad_x=None):
    global ep_reward, done, approach, approach_t, approach_x, approach_y
    global approach_ang, target_x, episode, info, mission_controller

    stop_alarm()

    if pinned_pad_x is not None:
        env.pin_pad(pinned_pad_x)

    obs, info = env.reset()   # info is now fresh: fuel=100, status=FLYING, steps=0

    ep_reward    = 0.0
    done         = False
    approach     = True
    approach_t   = 0.0
    approach_x   = -150.0
    approach_y   = 100.0
    approach_ang = -60.0
    target_x     = (env._pad_x1 + env._pad_x2) // 2
    dust_particles.clear()
    explosion_particles.clear()
    episode += 1

    if 'mission_controller' in globals() and mission_controller is not None:
        mission_controller.change_state(MissionState.LANDING)

    return obs, info


# ══════════════════════════════════════════════════════════════════════
# MAIN LOOP  —  initialise state
# ══════════════════════════════════════════════════════════════════════
MODES     = ["MANUAL", "AI", "PPO"]
mode_idx  = MODES.index(args.mode.upper()) if args.mode.upper() in MODES else 0
mode_name = MODES[mode_idx]

if mode_name == "PPO":
    try_load_ppo(args.model)

env = LunarLanderEnv(render_mode=None, randomize_pad=True, wind=False)
obs, info = env.reset()

mission_controller = MissionController()
if args.phase == "rover":
    mission_controller.change_state(MissionState.ROVER_EXPLORATION)
rover = LunarRover(start_x=280, start_y=1500)
rover_env = RoverEnv(world_width=3000, world_height=3000)
rover_autopilot = RoverAutopilot()          # rule-based fallback
rover_ppo = RoverPPOController()            # PPO-based autonomous controller
rover_telemetry = RoverTelemetryLogger()

episode      = 0
ep_reward    = 0.0
done         = False
approach     = False
approach_t   = 0.0
approach_x   = -150.0
approach_y   = 100.0
approach_ang = -60.0
target_x     = (env._pad_x1 + env._pad_x2) // 2

obs, info = do_reset(env)

if args.phase == "rover":
    mission_controller.change_state(MissionState.ROVER_EXPLORATION)

thrusting    = False
left_thrust  = False
right_thrust = False

lx, ly, lang = float(info.get("lander_x", WIDTH // 2)), \
               float(info.get("lander_y", 100)), 0.0

pinned_pad_x = None

running = True

while running:
    dt = clock.tick(60) / 1000.0
    dt = min(dt, 0.1)  # Cap dt to avoid massive physics jumps
    
    screen.fill((0, 0, 10))
    keys = pygame.key.get_pressed()

    # Update Mission Controller
    mission_controller.update(dt)

    # ── EVENTS ──────────────────────────────────────────────────────
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

            # Event handling depending on active phase
            if mission_controller.state == MissionState.ROVER_EXPLORATION:
                if event.key == pygame.K_r:
                    rover.reset()
                elif event.key == pygame.K_m:
                    rover.mode = "MANUAL" if rover.mode == "AUTONOMOUS" else "AUTONOMOUS"

            else:
                # Lander Events
                if coord_input_active:
                    if event.key == pygame.K_ESCAPE:
                        coord_input_active = False
                        coord_input_text   = ""
                        coord_input_error  = ""

                    elif event.key == pygame.K_BACKSPACE:
                        coord_input_text  = coord_input_text[:-1]
                        coord_input_error = ""

                    elif event.key == pygame.K_RETURN:
                        try:
                            val  = int(coord_input_text.strip())
                            half = PAD_HALF_WIDTH
                            lo   = half + 20
                            hi   = WIDTH - half - 20
                            if not (lo <= val <= hi):
                                coord_input_error = f"Out of range! Must be {lo}–{hi}"
                            else:
                                pinned_pad_x = val
                                obs, info    = do_reset(env, pinned_pad_x=val)
                                lx   = float(info.get("lander_x", WIDTH // 2))
                                ly   = float(info.get("lander_y", 100))
                                lang = 0.0
                                coord_input_active = False
                                coord_input_text   = ""
                                coord_input_error  = ""
                        except ValueError:
                            coord_input_error = "Enter a whole number (e.g. 400)"

                    elif event.unicode.isdigit():
                        coord_input_text += event.unicode

                else:
                    if event.key == pygame.K_r:
                        obs, info = do_reset(env, pinned_pad_x=pinned_pad_x)
                        lx   = float(info.get("lander_x", WIDTH // 2))
                        ly   = float(info.get("lander_y", 100))
                        lang = 0.0

                    elif event.key == pygame.K_m:
                        stop_alarm()
                        mode_idx  = (mode_idx + 1) % len(MODES)
                        mode_name = MODES[mode_idx]
                        if mode_name == "PPO" and ppo_agent is None:
                            try_load_ppo(args.model)

                    elif event.key == pygame.K_g:
                        show_graph = not show_graph

                    elif event.key == pygame.K_c:
                        coord_input_active = True
                        coord_input_text   = ""
                        coord_input_error  = ""

    # ── STATE PHYSICS / UPDATES ─────────────────────────────────────
    
    if mission_controller.state in (MissionState.LANDING, MissionState.LANDED, MissionState.MISSION_FAILED):
        # ── APPROACH ANIMATION ──────────────────────────────────────────
        if approach:
            approach_t += 0.0018
            progress = min(approach_t, 1.0)

            start_x, start_y = -220, HEIGHT * 0.72
            end_x,   end_y   = WIDTH * 0.38, HEIGHT * 0.10

            approach_x = start_x + (end_x - start_x) * (progress ** 1.15)
            approach_y = start_y - (start_y - end_y) * (1 - (1 - progress) ** 2.4)

            if   progress < 0.2:  approach_ang = -58
            elif progress < 0.4:  approach_ang = -42
            elif progress < 0.65: approach_ang = -28
            elif progress < 0.85: approach_ang = -12
            else:                 approach_ang = -4

            thrusting   = True
            left_thrust = right_thrust = False

            if progress >= 1.0:
                approach = thrusting = left_thrust = right_thrust = False

            lx, ly, lang = approach_x, approach_y, approach_ang

        # ── AGENT STEP ──────────────────────────────────────────────────
        elif not done:
            thrusting = left_thrust = right_thrust = False

            if coord_input_active:
                action = 0
            elif mode_name == "MANUAL":
                action = keyboard_action(keys)
            elif mode_name == "AI":
                action = rule_based_action(info, env._pad_x1, env._pad_x2)
            else:
                action = (ppo_agent.select_action(obs)
                          if ppo_agent is not None
                          else rule_based_action(info, env._pad_x1, env._pad_x2))

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

            thrusting    = (action == 1 and info.get("fuel", 0) > 0)
            left_thrust  = (action == 2)
            right_thrust = (action == 3)

            lx   = info.get("lander_x", lx)
            ly   = info.get("lander_y", ly)
            lang = info.get("angle",    lang)

            update_history(info.get("vy", 0), info.get("fuel", 0), ep_reward)

            if thrusting and ly > GROUND_Y - 80:
                spawn_dust(lx, GROUND_Y, info.get("vx", 0), info.get("vy", 0))

            if done:
                stop_alarm()
                status = info.get("status")
                if status == "LANDED":
                    mission_controller.change_state(MissionState.LANDED)
                elif status == "CRASHED":
                    spawn_explosion(lx, ly, info.get("vx", 0), info.get("vy", 0))
                    mission_controller.change_state(MissionState.MISSION_FAILED)

        # Draw Lander Scene
        draw_stars()
        draw_earth()
        draw_surface()
        draw_landing_zone(env._pad_x1, env._pad_x2)

        update_dust()
        draw_dust()

        # Alarm behavior
        if not approach and not done and info.get("status") == "FLYING":
            draw_warning_system(get_guidance_warnings(info))
        else:
            stop_alarm()

        if info.get("status") in ("FLYING", "LANDED") or approach:
            if thrusting:    draw_main_flame(lx, ly, lang)
            if left_thrust:  draw_side_flame(lx, ly, lang, "left")
            if right_thrust: draw_side_flame(lx, ly, lang, "right")

        if info.get("status") == "CRASHED" or mission_controller.state == MissionState.MISSION_FAILED:
            update_explosion()
            draw_explosion()

        draw_lander(lx, ly, lang)
        draw_hud(info, mode_name, episode, ep_reward)
        draw_controls_hint(mode_name)
        
        # Result panel
        if mission_controller.state == MissionState.LANDED:
            # Landing Confirmation Panel showing next phase prep
            txt = font_big.render("LANDED SAFELY!", True, (0, 255, 100))
            screen.blit(txt, (WIDTH // 2 - txt.get_width() // 2, HEIGHT // 2 - 80))
            summary = font.render(
                f"Episode Reward: {ep_reward:.1f}   |   Fuel Left: {info.get('fuel', 0.0):.1f}",
                True, (220, 220, 220)
            )
            screen.blit(summary, (WIDTH // 2 - summary.get_width() // 2, HEIGHT // 2 - 30))
            sub = font.render("Preparing next mission phase...", True, (0, 255, 100))
            screen.blit(sub, (WIDTH // 2 - sub.get_width() // 2, HEIGHT // 2 + 10))
        else:
            draw_result(info.get("status", ""), ep_reward, info.get("fuel", 0.0))

        if show_graph:
            draw_graph_panel()

        if coord_input_active:
            draw_coord_input()

    elif mission_controller.state == MissionState.ROVER_EXPLORATION:
        stop_alarm()
        
        # Control inputs
        if rover.mode == "MANUAL":
            forward_backward = 0.0
            turn_left_right = 0.0
            
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                rover.heading = 270.0
                forward_backward = 1.0
                rover.status = "EXPLORING (LEFT)"
            elif keys[pygame.K_d] or keys[pygame.K_RIGHT] or keys[pygame.K_SPACE]:
                rover.heading = 90.0
                forward_backward = 1.0
                rover.status = "EXPLORING (RIGHT)"
            elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
                forward_backward = -0.8
                rover.status = "REVERSING"
        else:
            # AUTONOMOUS mode: use PPO controller (with rule-based fallback)
            forward_backward, turn_left_right = rover_ppo.compute_controls(rover, rover_env, dt)

        # Update physics
        old_x, old_y = rover.x, rover.y
        rover.move(forward_backward, turn_left_right, dt)
        rover_env.check_collisions(rover, old_x, old_y)
        rover.update(dt)

        # Logging telemetry
        rover_telemetry.add_record(rover.get_telemetry())
        rover_telemetry.write_to_disk()

        # Render Rover interface
        draw_rover_scene(dt)

    # ── Cinematic transitions overlays ──
    mission_controller.draw_transition(screen, WIDTH, HEIGHT, font_big, font)

    if args.test_frames > 0:
        args.test_frames -= 1
        if args.test_frames == 0:
            if args.save_screenshot:
                pygame.image.save(screen, args.save_screenshot)
            running = False

    pygame.display.flip()

env.close()
pygame.quit()
sys.exit()