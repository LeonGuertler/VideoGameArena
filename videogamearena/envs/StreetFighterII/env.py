import os
import re
import time
import numpy as np
import retro  # Using gym-retro

class StreetFighterIIEnv:
    """A simplified environment for playing Street Fighter II: SCE with gym-retro."""
    
    # Button mapping for Sega Genesis in gym-retro (12 buttons total)
    BUTTONS = ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
    
    # Mapping from shorthand to button indices (simplified for Street Fighter II)
    BUTTON_MAP = {
        'u': 4,     # UP (jump)
        'd': 5,     # DOWN (crouch)
        'l': 6,     # LEFT
        'r': 7,     # RIGHT
        'lp': 0,    # Light Punch (B button)
        'mp': 1,    # Medium Punch (A button)
        'hp': 8,    # Heavy Punch (C button)
        'lk': 10,   # Light Kick (X button)
        'mk': 9,    # Medium Kick (Y button)
        'hk': 11,   # Heavy Kick (Z button)
        'o': 3,     # START
        'n': None,  # No operation
    }

    # Frame rate constants
    FPS = {
        'human': 60,
        'slow': 30,
        'super-slow': 10
    }

    def __init__(self, speed_mode='human'):
        """
        Initialize a new Street Fighter II: SCE environment using gym-retro.

        Args:
            speed_mode (str): 'human', 'slow', or 'super-slow'
        """
        if speed_mode not in self.FPS:
            raise ValueError(f"Speed mode must be one of {list(self.FPS.keys())}")
        
        self.env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')  # Load Street Fighter II
        print("Successfully loaded StreetFighterIISpecialChampionEdition-Genesis")
        
        self.speed_mode = speed_mode
        self.target_frame_time = 1.0 / self.FPS[speed_mode]
        self.last_frame_time = time.time()
        self.last_action_info = "No actions executed yet."
        self.total_reward = 0
        
        self.reset()
        self._skip_start_screen()

    def set_speed_mode(self, mode):
        """Change the game speed mode."""
        if mode not in self.FPS:
            raise ValueError(f"Speed mode must be one of {list(self.FPS.keys())}")
        self.speed_mode = mode
        self.target_frame_time = 1.0 / self.FPS[mode]
        print(f"Game speed set to {mode} mode ({self.FPS[mode]} FPS)")
    
    def _throttle_fps(self):
        """Throttle frame rate to maintain consistency."""
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
        """Convert an action string into button booleans for gym-retro."""
        action_groups = re.findall(r'\[(.[^\]]*)\]', action_string.lower())
        if not action_groups:
            self.last_action_info = "No actions executed."
            return [False] * len(self.BUTTONS)  # NOOP
        
        buttons = [False] * len(self.BUTTONS)
        for group in action_groups:
            for action in re.findall(r'[a-z]{1,2}', group):  # Match 1-2 letter actions (e.g., 'lp', 'r')
                if action in self.BUTTON_MAP and self.BUTTON_MAP[action] is not None:
                    buttons[self.BUTTON_MAP[action]] = True
        
        self.last_action_info = f"Executed action: {''.join(action_groups)}"
        return buttons

    def _skip_start_screen(self):
        """Skip the start screen by pressing start."""
        start_action = [False] * len(self.BUTTONS)
        start_action[self.BUTTON_MAP['o']] = True  # Press START button

        for _ in range(20):  # Repeat for 20 frames (longer start sequence)
            self.env.step(start_action)  # Press START
            self.env.step([False] * len(self.BUTTONS))  # No action (NOOP)
            time.sleep(0.1)  # Delay to simulate human input

    def step(self, action):
        """Step the environment with the given action."""
        if isinstance(action, str):
            action = self.parse_action_string(action)
        
        observation, reward, done, info = self.env.step(action)
        
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
        return self.env.reset()

    def render(self):
        """Render the environment."""
        self.env.render()

    def close(self):
        """Close the environment."""
        self.env.close()
        
    def get_action_instructions(self):
        """Return formatted action instructions."""
        return """
Action format: Submit actions in square brackets like [lp] or [r].
You can submit multiple actions simultaneously: [r] [lp] (equivalent to [rlp])

Available actions:
- [u]: Jump (Up)
- [d]: Crouch (Down)
- [l]: Move left
- [r]: Move right
- [lp]: Light Punch (B button)
- [mp]: Medium Punch (A button)
- [hp]: Heavy Punch (C button)
- [lk]: Light Kick (X button)
- [mk]: Medium Kick (Y button)
- [hk]: Heavy Kick (Z button)
- [o]: Start button
- [n]: No operation

Common combinations:
- [rlp]: Move right + Light Punch
- [dlk]: Crouch + Light Kick
- [uhp]: Jump + Heavy Punch
- [rmk]: Move right + Medium Kick
"""

# # Example usage
# if __name__ == "__main__":
#     env = StreetFighterIIEnv(speed_mode="slow")
    
#     obs = env.reset()
    
#     for _ in range(100):
#         obs, done, info = env.step("[rmk]")  # Move right + Medium Kick
#         env.render()
#         if done:
#             break
    
#     env.close()