import pygame

class MissionState:
    LANDING = "LANDING"
    LANDED = "LANDED"
    TRANSITION = "TRANSITION"
    ROVER_DEPLOYMENT = "ROVER_DEPLOYMENT"
    ROVER_EXPLORATION = "ROVER_EXPLORATION"
    MISSION_FAILED = "MISSION_FAILED"

class MissionController:
    def __init__(self):
        self.state = MissionState.LANDING
        self.state_timer = 0.0  # in seconds
        self.fade_alpha = 0
        self.deployment_messages = [
            "INITIALIZING SURFACE EXPLORATION...",
            "DEPLOYING AUTONOMOUS ROVER...",
            "ROVER SYSTEMS ONLINE"
        ]
        self.current_msg_idx = 0
        self.msg_timer = 0.0

    def change_state(self, new_state):
        self.state = new_state
        self.state_timer = 0.0
        self.msg_timer = 0.0
        self.current_msg_idx = 0
        if new_state == MissionState.TRANSITION:
            self.fade_alpha = 0
        elif new_state == MissionState.ROVER_DEPLOYMENT:
            self.fade_alpha = 255

    def update(self, dt):
        """dt: elapsed time in seconds since last frame"""
        self.state_timer += dt

        if self.state == MissionState.LANDED:
            # Show "LANDED SAFELY!" for 2.5 seconds, then transition
            if self.state_timer >= 2.5:
                self.change_state(MissionState.TRANSITION)

        elif self.state == MissionState.TRANSITION:
            # Fade to black over 1.5 seconds
            progress = min(self.state_timer / 1.5, 1.0)
            self.fade_alpha = int(progress * 255)
            if self.state_timer >= 1.5:
                self.change_state(MissionState.ROVER_DEPLOYMENT)

        elif self.state == MissionState.ROVER_DEPLOYMENT:
            # Play cinematic text sequence over 3.5 seconds
            self.msg_timer += dt
            if self.msg_timer < 1.2:
                self.current_msg_idx = 0
            elif self.msg_timer < 2.4:
                self.current_msg_idx = 1
            elif self.msg_timer < 3.6:
                self.current_msg_idx = 2
            else:
                self.change_state(MissionState.ROVER_EXPLORATION)

        elif self.state == MissionState.ROVER_EXPLORATION:
            # Slowly fade back in from black
            if self.fade_alpha > 0:
                self.fade_alpha = max(0, self.fade_alpha - int(255 * dt * 2))

    def draw_transition(self, screen, width, height, font_big, font_medium):
        """Draw fade overlays and deployment cinematic text"""
        # Draw fade overlay
        if self.state in (MissionState.TRANSITION, MissionState.ROVER_DEPLOYMENT, MissionState.ROVER_EXPLORATION):
            if self.fade_alpha > 0:
                fade_surface = pygame.Surface((width, height))
                fade_surface.fill((0, 0, 0))
                fade_surface.set_alpha(self.fade_alpha)
                screen.blit(fade_surface, (0, 0))

        # Draw deployment typewriter messages
        if self.state == MissionState.ROVER_DEPLOYMENT:
            screen.fill((0, 0, 0))  # Ensure background is black
            
            # Draw header
            header_text = font_big.render("MISSION SEQUENCE: ROVER DEPLOYMENT", True, (0, 255, 100))
            screen.blit(header_text, (width // 2 - header_text.get_width() // 2, height // 2 - 120))
            
            # Draw messages up to the current one
            for i in range(self.current_msg_idx + 1):
                msg = self.deployment_messages[i]
                color = (0, 200, 255) if i < self.current_msg_idx else (255, 255, 255)
                # If it's the last message ("ROVER SYSTEMS ONLINE"), make it green
                if msg == "ROVER SYSTEMS ONLINE":
                    color = (0, 255, 100)
                
                txt = font_medium.render(msg, True, color)
                screen.blit(txt, (width // 2 - txt.get_width() // 2, height // 2 - 20 + i * 40))

            # Draw a simulated system status bar at the bottom
            bar_width = 400
            bar_height = 10
            bar_x = width // 2 - bar_width // 2
            bar_y = height // 2 + 130
            pygame.draw.rect(screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height), border_radius=4)
            
            progress = min(self.msg_timer / 3.6, 1.0)
            pygame.draw.rect(screen, (0, 255, 100), (bar_x, bar_y, int(bar_width * progress), bar_height), border_radius=4)
