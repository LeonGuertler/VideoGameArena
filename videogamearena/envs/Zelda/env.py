import os
import re
import time
import numpy as np
import retro # as retro


class ZeldaEnv:
    """An environment for playing The Legend of Zelda: A Link to the Past with stable-retro."""
    
    # Button mapping for SNES controller
    BUTTONS = ['B', 'Y', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'X', 'L', 'R']
    
    # Mapping from our shorthand to button indices
    BUTTON_MAP = {
        'u': 4,      # UP
        'd': 5,      # DOWN
        'l': 6,      # LEFT
        'r': 7,      # RIGHT
        'a': 8,      # A button (action/interact)
        'b': 0,      # B button (sword)
        'x': 9,      # X button (item)
        'y': 1,      # Y button (dash)
        'L': 10,     # L button (map toggle)
        'R': 11,     # R button (usually not used)
        'o': 3,      # START (pause/menu)
        'p': 2,      # SELECT (inventory)
        'n': None,   # No operation
    }

    # Frame rate constants for different speed modes
    FPS = {
        'human': 60,       # 60 fps - normal SNES speed
        'slow': 30,        # 30 fps - half speed
        'super-slow': 10   # 10 fps - very slow for detailed analysis
    }

    def __init__(self, speed_mode='human', rom_path=None):
        """
        Initialize a new Zelda: A Link to the Past environment using stable-retro.

        Args:
            speed_mode (str): Game speed mode - 'human', 'slow', or 'super-slow'
            rom_path (str, optional): Path to the ROM file if you need to import it

        Returns:
            None
        """
        # Check if the speed mode is valid
        if speed_mode not in self.FPS:
            raise ValueError(f"Speed mode must be one of {list(self.FPS.keys())}")
        
        # Import ROM if provided
        if rom_path and os.path.exists(rom_path):
            try:
                self._import_rom(rom_path)
                print(f"Imported ROM from: {rom_path}")
            except Exception as e:
                print(f"Failed to import ROM: {str(e)}")
                print("Will attempt to use pre-existing ROM if available")
        
        # Create the stable-retro environment
        try:
            # Try with the exact name
            self.env = retro.make(game='TheLegendOfZelda-ALinkToThePast-Snes')
            print("Successfully loaded 'TheLegendOfZelda-ALinkToThePast-Snes'")
        except Exception as e1:
            # Try with alternative names
            try:
                zelda_games = [g for g in retro.data.list_games() if 'zelda' in g.lower()]
                if zelda_games:
                    print(f"Available Zelda games: {zelda_games}")
                    self.env = retro.make(game=zelda_games[0])
                    print(f"Successfully loaded '{zelda_games[0]}'")
                else:
                    raise ValueError("No Zelda games found. You need to import the ROM first.")
            except Exception as e2:
                print(f"Error loading game: {str(e1)}")
                print(f"Alternative method also failed: {str(e2)}")
                print("\nAvailable games:")
                for game in sorted(retro.data.list_games()):
                    print(f"  - {game}")
                raise RuntimeError("Could not load any Zelda game. See available games above.")
        
        # Setup frame rate control variables
        self.speed_mode = speed_mode
        self.target_frame_time = 1.0 / self.FPS[speed_mode]
        self.last_frame_time = time.time()

        # Setup UI tracking variables
        self.last_action_info = "No actions executed yet."
        self.total_reward = 0
        
        # Reset the environment
        self.reset()
        
        # Skip intro screens if needed
        self._skip_intro_screens()

    def _import_rom(self, rom_path):
        """Import a ROM into stable-retro."""
        # Check if the ROM exists
        if not os.path.exists(rom_path):
            raise FileNotFoundError(f"ROM file not found: {rom_path}")
        
        # Try to import using various methods
        if hasattr(retro.data, 'merge_rom'):
            retro.data.merge_rom(rom_path)
        else:
            # Alternative approach using Python's subprocess
            import subprocess
            import sys
            subprocess.run([
                sys.executable, "-m", "stable_retro.import_rom_py", rom_path
            ])

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

    def _skip_intro_screens(self):
        """Press start several times to skip intro screens."""
        # Press start button
        start_action = [False] * len(self.BUTTONS)
        start_action[self.BUTTON_MAP['o']] = True
        
        # Press A button
        a_action = [False] * len(self.BUTTONS)
        a_action[self.BUTTON_MAP['a']] = True
        
        # Press buttons in sequence to get past intro
        for _ in range(20):
            # Alternate between start, A, and nothing
            self.env.step(start_action)
            time.sleep(0.1)
            self.env.step([False] * len(self.BUTTONS))
            time.sleep(0.1)
            self.env.step(a_action)
            time.sleep(0.1)
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
        observation, reward, done, info = self.env.step(action)
        
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
        return self.env.render(mode=mode)

    def close(self):
        """Close the environment."""
        self.env.close()
        
    def get_action_instructions(self):
        """Return formatted instructions for the user on how to use actions."""
        return """
Action format: Submit actions in square brackets like [a] or [r].
You can submit multiple actions simultaneously: [r] [a] (equivalent to [ra])

Available actions for Zelda: A Link to the Past:
- [a]: A button (action/interact)
- [b]: B button (sword)
- [x]: X button (item)
- [y]: Y button (dash)
- [L]: L button (map toggle)
- [R]: R button
- [u]: Move up
- [d]: Move down
- [l]: Move left
- [r]: Move right
- [o]: START button (pause/menu)
- [p]: SELECT button (inventory)
- [n]: No operation

Common combinations:
- [rb]: Sword attack right
- [lb]: Sword attack left
- [ub]: Sword attack up
- [db]: Sword attack down
- [ry]: Dash right
- [ly]: Dash left
"""

    @staticmethod
    def list_available_games():
        """List all available games in stable-retro, with Zelda games highlighted."""
        try:
            all_games = retro.data.list_games()
            zelda_games = [game for game in all_games if 'zelda' in game.lower()]
            
            print(f"Found {len(all_games)} total games")
            if zelda_games:
                print(f"Found {len(zelda_games)} Zelda games:")
                for game in sorted(zelda_games):
                    print(f"  - {game}")
            else:
                print("No Zelda games found. You may need to import the ROM.")
                
            return zelda_games
        except Exception as e:
            print(f"Error listing games: {str(e)}")
            return []


# Example usage
if __name__ == "__main__":
    # List available Zelda games
    ZeldaALinkToPastEnv.list_available_games()
    
    # Create the environment
    # If you have the ROM file and need to import it:
    # env = ZeldaALinkToPastEnv(speed_mode="slow", rom_path="/path/to/zelda_alttp.sfc")
    
    # If the ROM is already imported:
    try:
        env = ZeldaALinkToPastEnv(speed_mode="slow")
        
        # Reset and get initial observation
        obs = env.reset()
        
        # Run for a few steps
        for _ in range(100):
            # Move right and press B (sword)
            obs, done, info = env.step("[rb]")
            
            # Break if the episode is done
            if done:
                break
        
        # Close the environment
        env.close()
    except Exception as e:
        print(f"Error creating environment: {str(e)}")
        print("\nYou might need to import the ROM first.")