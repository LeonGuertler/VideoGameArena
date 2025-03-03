import os
import re
import time
import numpy as np

# Import stable-retro
# import stable_retro as retro
import retro


class SuperMarioBrosEnv:
    """A simplified environment for playing Super Mario Bros with stable-retro."""
    
    # Button mapping for stable-retro
    BUTTONS = ['B', 'NULL', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
    
    # Mapping from our shorthand to button indices
    BUTTON_MAP = {
        'u': 4,     # UP
        'd': 5,     # DOWN
        'l': 6,     # LEFT
        'r': 7,     # RIGHT
        'a': 8,     # A button
        'b': 0,     # B button
        'o': 3,     # START
        'p': 2,     # SELECT
        'n': None,  # No operation
    }

    # Frame rate constants for different speed modes
    FPS = {
        'human': 60,       # 60 fps - normal NES speed
        'slow': 30,        # 30 fps - half speed
        'super-slow': 10   # 10 fps - very slow for detailed analysis
    }

    def __init__(self, speed_mode='human'):
        """
        Initialize a new Super Mario Bros environment using stable-retro.

        Args:
            speed_mode (str): Game speed mode - 'human', 'slow', or 'super-slow'

        Returns:
            None
        """
        # Check if the speed mode is valid
        if speed_mode not in self.FPS:
            raise ValueError(f"Speed mode must be one of {list(self.FPS.keys())}")
        
        # Create the stable-retro environment
        # self.env = retro.make(game='SuperMarioBros-Nes')
        self.env = retro.make(game='SuperMarioBros-Nes')
        print("Successfully loaded SuperMarioBros-Nes")
        
        # Setup frame rate control variables
        self.speed_mode = speed_mode
        self.target_frame_time = 1.0 / self.FPS[speed_mode]
        self.last_frame_time = time.time()

        # Setup UI tracking variables
        self.last_action_info = "No actions executed yet."
        self.total_reward = 0
        
        # Reset the environment
        self.reset()
        
        # Skip the start screen
        self._skip_start_screen()

    def set_speed_mode(self, mode):
        """Change the game speed mode."""
        if mode not in self.FPS:
            raise ValueError(f"Speed mode must be one of {list(self.FPS.keys())}")
        self.speed_mode = mode
        self.target_frame_time = 1.0 / self.FPS[mode]
        print(f"Game speed set to {mode} mode ({self.FPS[mode]} FPS)")
    
    def _throttle_fps(self):
        """Throttle the frame rate to maintain consistent speed."""
        # Calculate how long this frame took
        current_time = time.time()
        frame_execution_time = current_time - self.last_frame_time
        
        # Calculate the time to sleep to maintain the target FPS
        sleep_time = max(0, self.target_frame_time - frame_execution_time)
        
        # Sleep the appropriate amount to maintain constant frame rate
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        # Update the last frame time
        self.last_frame_time = time.time()
        
        return {
            "fps": self.FPS[self.speed_mode],
            "frame_time_ms": frame_execution_time * 1000,
            "sleep_time_ms": sleep_time * 1000
        }

    def parse_action_string(self, action_string):
        """
        Parse an action string into a list of booleans for stable-retro.
        
        Args:
            action_string (str): A string containing actions in [x] format
            
        Returns:
            list: Array of booleans for each button
        """
        # Extract actions using regex
        action_groups = re.findall(r'\[(.[^\]]*)\]', action_string.lower())
        
        if not action_groups:
            self.last_action_info = "No actions were executed because none were provided in the [x] format."
            return [False] * len(self.BUTTONS)  # NOOP
        
        # Initialize all buttons as not pressed
        buttons = [False] * len(self.BUTTONS)
        
        # Set the buttons that are pressed
        for group in action_groups:
            for char in group:
                if char in self.BUTTON_MAP and self.BUTTON_MAP[char] is not None:
                    buttons[self.BUTTON_MAP[char]] = True
        
        self.last_action_info = f"Executed action: {''.join(action_groups)}"
        return buttons

    def _skip_start_screen(self):
        """Press and release start to skip the start screen."""
        # Press start button
        start_action = [False] * len(self.BUTTONS)
        start_action[self.BUTTON_MAP['o']] = True
        
        # Press and release the start button a few times to get past intro screens
        for _ in range(10):
            self.env.step(start_action)
            self.env.step([False] * len(self.BUTTONS))
            time.sleep(0.1)

    def step(self, action):
        """
        Step the environment with the given action.
        
        Args:
            action: Either a list of booleans for each button, or
                   a string formatted like "[r][a]" to perform right+A
        
        Returns:
            A dictionary containing game state information
        """
        # If action is a string, parse it
        if isinstance(action, str):
            action = self.parse_action_string(action)
        
        # Perform the action in the environment
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        done = terminated or truncated
        # Use retro's built-in reward
        self.total_reward += reward
        
        # Add additional information to the info dict
        info.update({
            'total_reward': self.total_reward,
            'speed_mode': self.speed_mode,
            'last_action': self.last_action_info,
        })
        
        # Throttle the frame rate based on the selected speed mode
        fps_info = self._throttle_fps()
        info.update(fps_info)
        
        obs = {
            "visual": observation,
            "text": "Which buttons would you like to press?"
        }
        return obs, done, info

    def reset(self):
        """Reset the environment and return the initial observation."""
        # Reset the frame timer
        self.last_frame_time = time.time()
        
        # Reset tracking variables
        self.total_reward = 0
        self.last_action_info = "No actions executed yet."
        
        # Reset the stable-retro environment
        observation = self.env.reset()
        
        return observation

    def render(self, mode='human'):
        """Render the environment."""
        return self.env.render() #mode=mode)

    def close(self):
        """Close the environment."""
        self.env.close()
        
    def get_action_instructions(self):
        """Return formatted instructions for the user on how to use actions."""
        return """
Action format: Submit actions in square brackets like [a] or [r].
You can submit multiple actions simultaneously: [r] [a] (equivalent to [ra])

Available actions:
- [a]: Jump (A button)
- [b]: Run (B button)
- [u]: Move up (for climbing)
- [d]: Move down (for pipes)
- [l]: Move left
- [r]: Move right
- [o]: Start button
- [p]: Select button
- [n]: No operation

Common combinations:
- [ra]: Jump right
- [rb]: Run right
- [rab]: Jump while running right
- [la]: Jump left
- [lb]: Run left
"""


# # Example usage
# if __name__ == "__main__":
#     # Create the environment
#     env = SuperMarioBrosEnv(speed_mode="slow")
    
#     # Reset and get initial observation
#     obs = env.reset()
    
#     # Run for a few steps
#     for _ in range(100):
#         # Move right
#         obs, done, info = env.step("[r]")
        
#         # Break if the episode is done
#         if done:
#             break
    
#     # Close the environment
#     env.close()