import os
import re
import time
import numpy as np
import retro  # stable-retro

class StreetFighterIIEnv:
    """A simplified environment for Street Fighter II: Special Champion Edition with stable-retro."""

    # Button mapping for Genesis in stable-retro (12 buttons total)
    BUTTONS = ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']

    # Shorthand to button indices (based on Genesis six-button layout)
    BUTTON_MAP = {
        'u': 4,     # UP
        'd': 5,     # DOWN
        'l': 6,     # LEFT
        'r': 7,     # RIGHT
        'a': 1,     # A (light punch)
        'b': 0,     # B (medium punch)
        'c': 8,     # C (heavy punch)
        'x': 10,    # X (light kick)
        'y': 9,    # Y (medium kick)
        'z': 11,     # Z (heavy kick)
        's': 3,     # START
        'm': 2,     # MODE (unused in gameplay)
        'n': None,  # No operation
    }

    FPS = {
        'human': 60,       # 60 fps - normal Genesis speed
        'slow': 30,        # 30 fps - half speed
        'super-slow': 10   # 10 fps - for detailed analysis
    }

    def __init__(self, speed_mode='human'):
        """Initialize the Street Fighter II: Special Champion Edition environment."""
        if speed_mode not in self.FPS:
            raise ValueError(f"Speed mode must be one of {list(self.FPS.keys())}")

        # Load the Genesis game
        self.env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')
        print("Successfully loaded StreetFighterIISpecialChampionEdition-Genesis")

        # Frame rate control
        self.speed_mode = speed_mode
        self.target_frame_time = 1.0 / self.FPS[speed_mode]
        self.last_frame_time = time.time()

        # Tracking variables
        self.last_action_info = "No actions executed yet."
        self.total_reward = 0

        # Reset and skip title screen
        self.reset()
        self._skip_title_screen()

    def set_speed_mode(self, mode):
        """Change the game speed mode."""
        if mode not in self.FPS:
            raise ValueError(f"Speed mode must be one of {list(self.FPS.keys())}")
        self.speed_mode = mode
        self.target_frame_time = 1.0 / self.FPS[mode]
        print(f"Game speed set to {mode} mode ({self.FPS[mode]} FPS)")

    def _throttle_fps(self):
        """Throttle frame rate to maintain consistent speed."""
        current_time = time.time()
        frame_execution_time = current_time - self.last_frame_time
        sleep_time = max(0, self.target_frame_time - frame_execution_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        self.last_frame_time = time.time()
        return {
            "fps": self.FPS[self.speed_mode],
            "frame_time_ms": frame_execution_time * 1000,
            "sleep_time_ms": sleep_time * 1000
        }

    def parse_action_string(self, action_string):
        """Parse action string into button states."""
        action_groups = re.findall(r'\[(.[^\]]*)\]', action_string.lower())
        if not action_groups:
            self.last_action_info = "No actions executed (use [x] format)."
            return [False] * len(self.BUTTONS)  # NOOP

        buttons = [False] * len(self.BUTTONS)
        for group in action_groups:
            for char in group:
                if char in self.BUTTON_MAP and self.BUTTON_MAP[char] is not None:
                    buttons[self.BUTTON_MAP[char]] = True
        self.last_action_info = f"Executed action: {''.join(action_groups)}"
        return buttons

    def _skip_title_screen(self):
        """Skip the title screen by pressing Start."""
        start_action = [False] * len(self.BUTTONS)
        start_action[self.BUTTON_MAP['s']] = True
        for _ in range(10):  # Press Start a few times to bypass intro
            self.env.step(start_action)
            self.env.step([False] * len(self.BUTTONS))
            time.sleep(0.1)

    def step(self, action):
        """Step the environment with the given action."""
        if isinstance(action, str):
            action = self.parse_action_string(action)

        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.total_reward += reward

        info.update({
            'total_reward': self.total_reward,
            'speed_mode': self.speed_mode,
            'last_action': self.last_action_info,
        })
        fps_info = self._throttle_fps()
        info.update(fps_info)

        obs = {
            "visual": observation,
            "text": "Which buttons would you like to press?"
        }
        return obs, done, info

    def reset(self):
        """Reset the environment."""
        self.last_frame_time = time.time()
        self.total_reward = 0
        self.last_action_info = "No actions executed yet."
        observation = self.env.reset()
        return observation

    def render(self):
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Close the environment."""
        print("Total Rewards:", self.total_reward)
        self.env.close()

    def get_action_instructions(self):
        """Return instructions for user actions."""
        return """
Action format: Use [x] format, e.g., [r][a] or [ra] for simultaneous inputs.

Available actions:
- [a]: Light Punch (A)
- [b]: Medium Punch (B)
- [c]: Heavy Punch (C)
- [x]: Light Kick (X)
- [y]: Medium Kick (Y)
- [z]: Heavy Kick (Z)
- [u]: Up
- [d]: Down
- [l]: Left
- [r]: Right
- [s]: Start
- [m]: Mode (unused)
- [n]: No operation

Common combinations:
- [ra]: Move right + light punch
- [dz]: Crouch + heavy kick (sweep)
- [ub]: Jump + medium punch
"""

# # Example usage
# if __name__ == "__main__":
#     env = StreetFighterIISCEEnv(speed_mode="slow")
#     obs = env.reset()
#     for _ in range(100):
#         obs, done, info = env.step("[r][a]")  # Move right + light punch
#         env.render()
#         if done:
#             break
#     env.close()