import random
import math
import pygame

class RoverEnv:
    def __init__(self, world_width=3000, world_height=3000):
        self.world_width = world_width
        self.world_height = world_height
        
        self.rocks = []
        self.craters = []
        self.hazards = []
        self.cam_stars = []
        self._font_cam = None
        self.generate_terrain()
        self._init_cam_stars()

    def _init_cam_stars(self):
        rng = random.Random(999)
        self.cam_stars = [
            (rng.randint(0, 300), rng.randint(0, 60), rng.randint(1, 2), rng.randint(140, 240))
            for _ in range(30)
        ]

    def _get_cam_font(self):
        if self._font_cam is None:
            if not pygame.font.get_init():
                pygame.font.init()
            self._font_cam = pygame.font.SysFont("Consolas", 10)
        return self._font_cam

    def generate_terrain(self):
        random.seed(42)  # For reproducible terrain

        ROVER_HALF_W   = 55.0   # half the rover sprite width — sets collision clearance
        MIN_ROCK_GAP   = 200.0  # minimum clear gap between any two rocks (px)
        MIN_CRATER_GAP = 160.0  # minimum clear gap between craters and other obstacles (px)

        # ── Rocks ──────────────────────────────────────────────────────
        # Keep landing zone clear: obstacles spawn starting at x = 500
        curr_x = 500.0
        all_obstacles = []   # shared list to cross-check spacing
        for i in range(22):  # fewer rocks but better spaced
            curr_x += random.uniform(200, 320)  # wide gap between placements
            if curr_x > self.world_width - 100:
                break
            ry   = 1500.0 + random.uniform(-18, 18)   # tighter Y so rover always hits them
            size = random.uniform(18, 32)

            # Reject if too close to any already-placed obstacle
            too_close = any(
                abs(curr_x - ox) < (size + osize + MIN_ROCK_GAP)
                for ox, _, osize in all_obstacles
            )
            if too_close:
                curr_x += MIN_ROCK_GAP  # push further right and skip
                continue

            # Pre-generate boulder facet points (relative offsets) for consistent 3D rock shape
            num_facets = 7
            facet_angles = [j * (2 * math.pi / num_facets) for j in range(num_facets)]
            offsets = [random.uniform(0.75, 1.25) for _ in range(num_facets)]
            facets = [(math.cos(a) * offsets[k], math.sin(a) * offsets[k])
                      for k, a in enumerate(facet_angles)]

            self.rocks.append({
                "id": f"R-{i+1:02d}",
                "x": curr_x,
                "y": ry,
                "r": size,
                "scanned": False,
                "facets": facets,
                "tint": random.randint(120, 155)
            })
            all_obstacles.append((curr_x, ry, size))

        # ── Craters ────────────────────────────────────────────────────
        curr_cx = 600.0
        for j in range(14):  # fewer craters, well separated
            curr_cx += random.uniform(240, 380)
            if curr_cx > self.world_width - 120:
                break
            cy   = 1500.0 + random.uniform(-20, 20)
            size = random.uniform(32, 58)

            # Reject if overlapping any rock or previous crater
            too_close = any(
                abs(curr_cx - ox) < (size + osize + MIN_CRATER_GAP)
                for ox, _, osize in all_obstacles
            )
            if too_close:
                curr_cx += MIN_CRATER_GAP
                continue

            self.craters.append({
                "id": f"C-{j+1:02d}",
                "x": curr_cx,
                "y": cy,
                "r": size,
                "scanned": False
            })
            all_obstacles.append((curr_cx, cy, size))

        # ── Hazard patches (regolith drifts) — placed between obstacles ──
        curr_hx = 700.0
        for _ in range(12):
            curr_hx += random.uniform(200, 310)
            if curr_hx > self.world_width - 180:
                break
            hy   = 1500.0 + random.uniform(-20, 20)
            size = random.uniform(60, 100)
            self.hazards.append({"x": curr_hx, "y": hy, "r": size})

        # Sort obstacles by X coordinate
        self.rocks.sort(key=lambda o: o["x"])
        self.craters.sort(key=lambda o: o["x"])

    def check_collisions(self, rover, old_x, old_y):
        """Perform boundary checks and obstacle/crater collisions."""
        # World boundary check
        min_margin = getattr(rover, "min_x", 100.0)
        if rover.x < min_margin:
            rover.x = min_margin
            rover.speed = 0.0
        elif rover.x > self.world_width - 30.0:
            rover.x = self.world_width - 30.0
            rover.speed = 0.0

        # Scan for proximity to scan rocks/craters
        for rock in self.rocks:
            dist = math.hypot(rover.x - rock["x"], rover.y - rock["y"])
            if dist < 85 and not rock["scanned"]:
                rock["scanned"] = True
                rover.scanned_rocks += 1

        for crater in self.craters:
            dist = math.hypot(rover.x - crater["x"], rover.y - crater["y"])
            if dist < 120 and not crater["scanned"]:
                crater["scanned"] = True
                rover.scanned_craters += 1

    def get_hazard_score(self, rover):
        """Calculate local hazard index (0.0 to 1.0) based on nearby obstacles."""
        max_dist = 140.0
        min_hazard_dist = 1000.0
        
        # Check obstacles ahead of rover
        for obstacle in self.rocks + self.craters:
            dx = obstacle["x"] - rover.x
            if dx > -20.0:
                dist = math.hypot(dx, obstacle["y"] - rover.y) - obstacle["r"]
                if dist < min_hazard_dist:
                    min_hazard_dist = max(0.0, dist)
                
        if min_hazard_dist < max_dist:
            return round((max_dist - min_hazard_dist) / max_dist, 2)
        return 0.0

    def get_terrain_type(self, rover):
        """Identify terrain style under the rover."""
        for hz in self.hazards:
            dist = math.hypot(rover.x - hz["x"], rover.y - hz["y"])
            if dist < hz["r"]:
                return "REGOLITH DRIFT (HAZARDOUS)"
        for crater in self.craters:
            dist = math.hypot(rover.x - crater["x"], rover.y - crater["y"])
            if dist < crater["r"] + 25:
                return "CRATER RIM"
        return "BASALTIC PLAINS"

    def get_camera_frame(self, rover, width=200, height=120):
        """
        Capture the live first-person optical feed from the rover's forward Hazcam.
        Renders true 3D perspective terrain, approaching boulders/craters,
        and an AI Machine Vision overlay with dynamic object detection tags.
        """
        frame = pygame.Surface((width, height))
        
        # ── 1. SKY & HORIZON ──
        horizon_y = int(height * 0.44)  # approx line 52
        
        # Space black with subtle celestial gradient
        frame.fill((4, 5, 12))
        
        # Distant stars in camera FOV
        for sx, sy, s_sz, s_b in self.cam_stars:
            if sy < horizon_y - 2:
                frame.set_at((sx % width, sy), (s_b, s_b, s_b))
        
        # Faint lunar horizon glow
        pygame.draw.line(frame, (35, 45, 60), (0, horizon_y), (width, horizon_y), 1)

        # ── 2. LUNAR REGOLITH GROUND (PERSPECTIVE) ──
        # Draw receding vertical gradient for lunar ground
        ground_height = height - horizon_y
        for y_step in range(ground_height):
            t = y_step / max(1, ground_height)
            # Regolith color deepens as it nears the rover
            col_val = int(28 + t * 48)
            line_y = horizon_y + y_step
            pygame.draw.line(frame, (col_val, col_val + 2, col_val + 6), (0, line_y), (width, line_y))

        # Subtle perspective terrain lines receding to horizon vanishing point
        center_x = width // 2
        for offset_x in (-70, -35, 0, 35, 70):
            p_bottom_x = center_x + offset_x * 2.2
            pygame.draw.line(frame, (38, 42, 52), (center_x + offset_x * 0.2, horizon_y), (p_bottom_x, height), 1)

        # ── 3. PERSPECTIVE OBSTACLES IN FRONT ──
        # Camera looks strictly forward along current heading (90 deg / East or 270 deg / West)
        max_view_dist = 260.0
        focal_length = 75.0
        
        is_facing_left = (rover.heading == 270.0)
        visible_items = []
        
        for obs in self.rocks:
            dx = (rover.x - obs["x"]) if is_facing_left else (obs["x"] - rover.x)
            if 0 < dx < max_view_dist:
                dy = (rover.y - obs["y"]) if is_facing_left else (obs["y"] - rover.y)
                visible_items.append(("rock", obs, dx, dy))

        for obs in self.craters:
            dx = (rover.x - obs["x"]) if is_facing_left else (obs["x"] - rover.x)
            if 0 < dx < max_view_dist:
                dy = (rover.y - obs["y"]) if is_facing_left else (obs["y"] - rover.y)
                visible_items.append(("crater", obs, dx, dy))

        # ── Regolith / hazard patches ──
        for obs in self.hazards:
            dx = (rover.x - obs["x"]) if is_facing_left else (obs["x"] - rover.x)
            if -obs["r"] < dx < max_view_dist:
                dx = max(1.0, dx)  # avoid zero-division
                dy = (rover.y - obs["y"]) if is_facing_left else (obs["y"] - rover.y)
                visible_items.append(("hazard", obs, dx, dy))

        # Sort items by distance (furthest to nearest) for correct depth ordering
        visible_items.sort(key=lambda item: item[2], reverse=True)

        font = self._get_cam_font()
        hud_targets = []

        for item_type, obs, dx, dy in visible_items:
            # Perspective math: scale factor inversely proportional to distance
            scale = focal_length / (dx + focal_length)
            
            # Map lateral offset to camera X
            cam_x = int(center_x + (dy / (dx * 0.5 + 25.0)) * (width * 0.5))
            
            # Distance mapping to screen Y (horizon to bottom)
            cam_y = int(horizon_y + (1.0 - scale ** 0.85) * (height - horizon_y))
            
            # Clamp inside frame margin
            if not (-40 <= cam_x <= width + 40 and horizon_y - 10 <= cam_y <= height + 30):
                continue

            if item_type == "rock":
                # Render 3D shaded boulder
                r_screen = max(2.5, obs["r"] * scale * 1.5)
                
                # Ground contact shadow underneath
                shadow_rect = pygame.Rect(
                    int(cam_x - r_screen * 1.1),
                    int(cam_y + r_screen * 0.3),
                    int(r_screen * 2.2),
                    int(r_screen * 0.6)
                )
                pygame.draw.ellipse(frame, (12, 14, 20), shadow_rect)

                # Shaded boulder facets
                tint = obs.get("tint", 135)
                body_color = (tint, tint, tint + 5)
                sunlit_color = (min(255, tint + 55), min(255, tint + 55), min(255, tint + 60))
                shadow_color = (max(20, tint - 60), max(20, tint - 60), max(25, tint - 55))

                poly_pts = []
                for fx, fy in obs["facets"]:
                    poly_pts.append((int(cam_x + fx * r_screen), int(cam_y + fy * r_screen * 0.85)))
                
                if len(poly_pts) >= 3:
                    # Body
                    pygame.draw.polygon(frame, body_color, poly_pts)
                    # Shadow side (lower-right half)
                    shadow_pts = [poly_pts[0], poly_pts[1], poly_pts[2], (cam_x, cam_y)]
                    pygame.draw.polygon(frame, shadow_color, shadow_pts)
                    # Sunlit facet (top-left)
                    highlight_pts = [poly_pts[4], poly_pts[5], poly_pts[6], (cam_x, cam_y)]
                    pygame.draw.polygon(frame, sunlit_color, highlight_pts)
                    # Outline
                    pygame.draw.polygon(frame, (25, 28, 35), poly_pts, 1)

                # Collect bounding box for AI target HUD
                bbox_w = int(r_screen * 2.4) + 4
                bbox_h = int(r_screen * 1.8) + 4
                hud_targets.append({
                    "rect": pygame.Rect(cam_x - bbox_w // 2, cam_y - bbox_h // 2, bbox_w, bbox_h),
                    "dist": dx,
                    "label": f"BOULDER {dx/10.0:.1f}m",
                    "scanned": obs.get("scanned", False),
                    "type": "rock"
                })

            elif item_type == "crater":
                # Render perspective crater depression
                r_screen_w = max(6.0, obs["r"] * scale * 1.4)
                r_screen_h = max(2.5, r_screen_w * 0.38)
                
                crater_rect = pygame.Rect(
                    int(cam_x - r_screen_w),
                    int(cam_y - r_screen_h),
                    int(r_screen_w * 2),
                    int(r_screen_h * 2)
                )
                
                # Dark interior bowl
                pygame.draw.ellipse(frame, (14, 15, 22), crater_rect)
                # Raised illuminated rim (top/left edge)
                pygame.draw.arc(frame, (150, 155, 170), crater_rect, 0.2, 3.14, 2)
                # Shaded inner rim
                pygame.draw.arc(frame, (35, 38, 48), crater_rect, 3.14, 6.28, 1)

                bbox_w = int(r_screen_w * 2.1) + 4
                bbox_h = int(r_screen_h * 2.1) + 4
                hud_targets.append({
                    "rect": pygame.Rect(cam_x - bbox_w // 2, cam_y - bbox_h // 2, bbox_w, bbox_h),
                    "dist": dx,
                    "label": f"CRATER {dx/10.0:.1f}m",
                    "scanned": obs.get("scanned", False),
                    "type": "crater"
                })

            elif item_type == "hazard":
                # Render regolith drift patch as a ground-level discoloured zone
                r_screen_w = max(8.0, obs["r"] * scale * 1.6)
                r_screen_h = max(3.0, r_screen_w * 0.22)
                patch_y    = min(height - 2, cam_y + int(r_screen_w * 0.1))

                patch_rect = pygame.Rect(
                    int(cam_x - r_screen_w),
                    int(patch_y - r_screen_h),
                    int(r_screen_w * 2),
                    int(r_screen_h * 2),
                )
                # Amber-tinted soft patch surface
                pygame.draw.ellipse(frame, (55, 42, 18), patch_rect)
                # Subtle grain lines inside
                for grain_i in range(3):
                    gx = int(cam_x - r_screen_w * 0.6 + grain_i * r_screen_w * 0.6)
                    pygame.draw.line(
                        frame, (70, 55, 25),
                        (gx, patch_y),
                        (gx + int(r_screen_w * 0.2), int(patch_y - r_screen_h * 0.5)), 1
                    )

                bbox_w = int(r_screen_w * 2.0) + 4
                bbox_h = int(r_screen_h * 3.0) + 4
                hud_targets.append({
                    "rect": pygame.Rect(cam_x - bbox_w // 2, patch_y - bbox_h // 2, bbox_w, bbox_h),
                    "dist": dx,
                    "label": f"REGOLITH {dx/10.0:.1f}m",
                    "scanned": False,
                    "type": "hazard"
                })

        # Subtle scanline effect on landscape
        for sl_y in range(0, height, 4):
            pygame.draw.line(frame, (12, 14, 20), (0, sl_y), (width, sl_y), 1)

        # ── 4. MACHINE VISION AI HUD OVERLAY ──
        # Render AI object detection bounding brackets for closest obstacles
        for target in hud_targets[:4]:  # prioritize 4 nearest obstacles
            trect = target["rect"]
            dist_val = target["dist"]
            t_type   = target.get("type", "rock")

            # Determine detection alert color based on type and proximity
            if t_type == "hazard":
                # Regolith drift: amber/orange regardless of distance
                box_color = (255, 165, 0) if (pygame.time.get_ticks() // 400) % 2 == 0 else (200, 120, 0)
            elif dist_val < 55.0:
                # Critical collision danger - flashing red/orange
                box_color = (255, 60, 60) if (pygame.time.get_ticks() // 200) % 2 == 0 else (255, 140, 0)
            elif dist_val < 110.0:
                # Approaching hazard / scanning - caution yellow
                box_color = (255, 200, 40)
            else:
                # Safe distant target - high-tech cyan/green
                box_color = (0, 240, 160)

            # Draw sci-fi corner brackets instead of solid box
            bx, by, bw, bh = trect.x, trect.y, trect.width, trect.height
            c_len = min(6, bw // 3, bh // 3)
            if c_len > 1:
                # Top-left
                pygame.draw.line(frame, box_color, (bx, by), (bx + c_len, by), 1)
                pygame.draw.line(frame, box_color, (bx, by), (bx, by + c_len), 1)
                # Top-right
                pygame.draw.line(frame, box_color, (bx + bw, by), (bx + bw - c_len, by), 1)
                pygame.draw.line(frame, box_color, (bx + bw, by), (bx + bw, by + c_len), 1)
                # Bottom-left
                pygame.draw.line(frame, box_color, (bx, by + bh), (bx + c_len, by + bh), 1)
                pygame.draw.line(frame, box_color, (bx, by + bh), (bx, by + bh - c_len), 1)
                # Bottom-right
                pygame.draw.line(frame, box_color, (bx + bw, by + bh), (bx + bw - c_len, by + bh), 1)
                pygame.draw.line(frame, box_color, (bx + bw, by + bh), (bx + bw, by + bh - c_len), 1)

            # Text tag with dark background for maximum readability
            tag_surf = font.render(target["label"], True, box_color)
            tag_y = max(horizon_y + 1, by - 12)
            tag_x = max(2, min(width - tag_surf.get_width() - 4, bx))
            tag_bg = pygame.Surface((tag_surf.get_width() + 4, tag_surf.get_height() + 2))
            tag_bg.fill((8, 10, 16))
            frame.blit(tag_bg, (tag_x - 2, tag_y - 1))
            frame.blit(tag_surf, (tag_x, tag_y))

        # ── 5. OPTICAL CAMERA HUD & TELEMETRY OVERLAY ──
        # Center target reticle / crosshair
        pygame.draw.line(frame, (0, 180, 220), (center_x - 8, horizon_y), (center_x + 8, horizon_y), 1)
        pygame.draw.line(frame, (0, 180, 220), (center_x, horizon_y - 5), (center_x, horizon_y + 5), 1)

        # Top Bar: Camera ID and Recording Indicator
        top_bar = pygame.Surface((width - 4, 15))
        top_bar.fill((8, 10, 16))
        frame.blit(top_bar, (2, 2))
        
        cam_id_lbl = font.render("CAM-01 [FWD HAZCAM]", True, (0, 220, 255))
        frame.blit(cam_id_lbl, (6, 4))

        # Blinking REC dot
        if (pygame.time.get_ticks() // 500) % 2 == 0:
            pygame.draw.circle(frame, (255, 50, 50), (width - 36, 9), 3)
        rec_lbl = font.render("REC", True, (255, 80, 80))
        frame.blit(rec_lbl, (width - 28, 4))

        # Bottom Proximity / Status Bar
        if hud_targets:
            nearest = min(hud_targets, key=lambda t: t["dist"])
            if nearest["dist"] < 55.0:
                stat_str = f"! COLLISION ALERT: {nearest['dist']/10.0:.1f}m !"
                stat_col = (255, 60, 60)
            elif nearest["dist"] < 110.0:
                stat_str = f"PROX HAZARD: {nearest['dist']/10.0:.1f}m"
                stat_col = (255, 190, 30)
            else:
                stat_str = f"TARGET LOCKED: {nearest['dist']/10.0:.1f}m"
                stat_col = (0, 230, 160)
        else:
            dir_str = "WEST 270°" if (rover.heading == 270.0) else "EAST 90°"
            stat_str = f"PATH CLEAR // {dir_str}"
            stat_col = (0, 200, 140)

        # Semi-transparent bottom HUD bar
        stat_surf = font.render(stat_str, True, stat_col)
        hud_bg = pygame.Surface((width - 8, 14))
        hud_bg.fill((8, 10, 16))
        frame.blit(hud_bg, (4, height - 17))
        frame.blit(stat_surf, (7, height - 16))

        # Outer camera border
        pygame.draw.rect(frame, (0, 160, 220), (0, 0, width, height), 1)

        return frame

