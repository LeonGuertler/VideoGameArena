# import os 
# from typing import Optional



# class SuperMarioEnv(vga.Env):
#     """ TODO """
#     def __init__(self):
#         self.env = retro.make("Super_mario_brothers")

#         self.key_mapping = {}

#     def reset(self, seed: Optional[int]=None):


#     def step(self, action)


# import os
# from nes_py import NESEnv

# # Define action spaces
# COMPLEX_MOVEMENT = [
#     ['NOOP'],
#     ['right'],
#     ['right', 'A'],
#     ['right', 'B'],
#     ['right', 'A', 'B'],
#     ['A'],
#     ['left'],
#     ['left', 'A'],
#     ['left', 'B'],
#     ['left', 'A', 'B'],
#     ['down'],
#     ['up'],
# ]

# class SuperMarioBrosEnv(vga.NESEnvBaseClass):
#     def __init__(self):
#         rom_path = os.path.join("SuperMarioBros", "super-mario-bros.nes")
#         super().__init__(rom_path)

#     def reset(self) -> vga.Observation:
#         """ TODO """


#         # this should restart the env, reset necessary parameters and return the player prompt (as a string; explaining how to play the game)
#         # the observation should include both the current frames as well as a string, explaining the rules and objective of the game 

#     def step(self, action) -> Tuple[vga.Observation, bool, vga.Info]: # return the next observation (frames + relevant information as string), done and Info (empty dict)



#     def close(self):
#         # this should return the reward for the player






# class MiniMarioEnv(vga.NESEnvBaseClass):
#     """A minimal Super Mario Bros environment."""
    
#     def __init__(self):



#         # Path to your Super Mario Bros ROM file
#         rom_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'super-mario-bros.nes')
#         super(MiniMarioEnv, self).__init__(rom_path)
        
#         # Initialize variables
#         self._x_position_last = 0
#         self._time_last = 0
        
#         # Reset the environment
#         self.reset()
#         # Skip the start screen
#         self._skip_start_screen()
#         # Create a backup state
#         self._backup()
    
#     def _skip_start_screen(self):
#         """Press and release start to skip the start screen."""
#         # Press and release the start button
#         self._frame_advance(8)  # 8 corresponds to the START button
#         self._frame_advance(0)
#         # Wait until the game starts
#         while self.ram[0x07f8:0x07fb].sum() == 0:  # Check if time is shown
#             self._frame_advance(8)
#             self._frame_advance(0)
    
#     def _get_reward(self):
#         """Return a simple reward based on x-position progress."""
#         # Calculate x position reward
#         x_position = self.ram[0x6d] * 0x100 + self.ram[0x86]
#         x_reward = x_position - self._x_position_last
#         self._x_position_last = x_position
        
#         # Calculate time penalty
#         time = int(''.join(map(str, self.ram[0x07f8:0x07fb])))
#         time_penalty = time - self._time_last
#         self._time_last = time
        
#         # Calculate death penalty
#         player_state = self.ram[0x000e]
#         death_penalty = -15 if player_state == 0x0b or player_state == 0x06 else 0
        
#         return x_reward + time_penalty + death_penalty
    
#     def _get_done(self):
#         """Return True if the episode is over."""
#         # Check if Mario is dead or dying
#         player_state = self.ram[0x000e]
#         return player_state == 0x0b or player_state == 0x06 or self.ram[0x075a] == 0xff
    
#     def _get_info(self):
#         """Return information about the current state."""
#         return {
#             'x_pos': self.ram[0x6d] * 0x100 + self.ram[0x86],
#             'y_pos': 255 - self.ram[0x03b8],
#             'world': self.ram[0x075f] + 1,
#             'stage': self.ram[0x075c] + 1,
#             'time': int(''.join(map(str, self.ram[0x07f8:0x07fb]))),
#             'coins': int(''.join(map(str, self.ram[0x07ed:0x07ef]))),
#             'score': int(''.join(map(str, self.ram[0x07de:0x07e4]))),
#             'life': self.ram[0x075a]
#         }

# # Function to create and register the environment
# def register_mario_env():
#     gym.envs.registration.register(
#         id='MiniMario-v0',
#         entry_point='mini_mario:MiniMarioEnv',
#         max_episode_steps=9999999,
#         nondeterministic=True,
#     )

# # Alias to gym.make for convenience
# def make(env_id):
#     return gym.make(env_id)


import os, re 
from nes_py import NESEnv

class SuperMarioBrosEnv(NESEnv):
    """A minimal Super Mario Bros environment optimized for language model interaction."""
    
    # Define the action mapping for language models
    ACTION_MAPPING = {
        'a': 'A',           # Jump
        'b': 'B',           # Run
        'u': 'up',          # Move up (for climbing)
        'd': 'down',        # Move down (for pipes)
        'l': 'left',        # Move left
        'r': 'right',       # Move right
        'n': 'NOOP',        # No operation
    }
    
    def __init__(self):
        # Path to your Super Mario Bros ROM file
        rom_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'super-mario-bros.nes')
        super().__init__(rom_path)
        
        # Initialize variables
        self._x_position_last = 0
        self._time_last = 0
        self.last_action_info = "No actions executed yet."
        
        # Reset the environment
        self.reset()
        # Skip the start screen
        self._skip_start_screen()
        # Create a backup state
        self._backup()
    
    # def _skip_start_screen(self):
    #     """Press and release start to skip the start screen."""
    #     # Press and release the start button
    #     self._frame_advance(8)  # 8 corresponds to the START button
    #     self._frame_advance(0)
    #     # Wait until the game starts
    #     while self.ram[0x07f8:0x07fb].sum() == 0:  # Check if time is shown
    #         self._frame_advance(8)
    #         self._frame_advance(0)
    
    # def _skip_start_screen(self):
    #     """Press and release start to skip the start screen and navigate through the initial menus."""
    #     # Wait a few frames first to make sure the ROM is loaded
    #     for _ in range(5):
    #         self._frame_advance(0)
        
    #     # Press start repeatedly with some delay until the game starts
    #     for _ in range(20):  # Try more times to ensure we get past the start screen
    #         # Press start
    #         self._frame_advance(8)  # START button (bit 4)
    #         # Wait a few frames
    #         for _ in range(3):
    #             self._frame_advance(0)
    #         # Release start
    #         self._frame_advance(0)
            
    #         # Check if the game has started by looking for the time display
    #         # or Mario's position being initialized
    #         if (self.ram[0x07f8:0x07fb].sum() > 0 or  # Time is displayed
    #             (self.ram[0x6d] > 0 or self.ram[0x86] > 0)):  # Mario has a position
    #             break
            
    #         # Add a small delay between attempts
    #         for _ in range(5):
    #             self._frame_advance(0)
        
    #     # One final check - if we still haven't started, try one more approach
    #     if self.ram[0x07f8:0x07fb].sum() == 0:
    #         # Sometimes we need to press A or B to get past certain screens
    #         for button in [0x80, 0x40, 0x10, 0x80]:  # Try A, B, START, A
    #             self._frame_advance(button)
    #             for _ in range(3):
    #                 self._frame_advance(0)
        
    #     # Final waiting period to ensure the game is fully loaded
    #     for _ in range(10):
    #         self._frame_advance(0)
        
    #     # Update the position and time trackers
    #     self._x_position_last = self.ram[0x6d] * 0x100 + self.ram[0x86]
    #     self._time_last = int(''.join(map(str, filter(lambda x: x != 0, self.ram[0x07f8:0x07fb]))) or '0')

    def _skip_start_screen(self):
        """Press and release start to skip all intro screens and get to actual gameplay."""
        # Wait a few frames first to make sure the ROM is loaded
        for _ in range(5):
            self._frame_advance(0)
        
        # Press start repeatedly with some delay until we get past the title screen
        for _ in range(20):
            # Press start
            self._frame_advance(8)  # START button (bit 4)
            # Wait a few frames
            for _ in range(3):
                self._frame_advance(0)
            # Release start
            self._frame_advance(0)
            
            # Check if the game has moved past the title screen
            if (self.ram[0x075f] > 0 or self.ram[0x075c] > 0):  # World or stage values are set
                break
            
            # Add a small delay between attempts
            for _ in range(5):
                self._frame_advance(0)
        
        # Now we need to get past the black screen with the world announcement
        # This screen shows "WORLD 1-1" and "MARIO × 3"
        # Press start and wait until Mario's x position is initialized
        for _ in range(30):  # Try multiple times
            # Press start
            self._frame_advance(8)
            # Wait a few frames
            for _ in range(3):
                self._frame_advance(0)
            # Release start
            self._frame_advance(0)
            
            # Check if Mario's position has been initialized to something other than zero
            # When actual gameplay starts, Mario will have a non-zero x position
            if self.ram[0x86] > 20:  # Mario has a starting position (typically around 40)
                break
                
            # Add a small delay between attempts
            for _ in range(5):
                self._frame_advance(0)
        
        # Final waiting period to ensure the game is fully loaded
        for _ in range(10):
            self._frame_advance(0)
        
        # Update the position and time trackers
        self._x_position_last = self.ram[0x6d] * 0x100 + self.ram[0x86]
        self._time_last = int(''.join(map(str, filter(lambda x: x != 0, self.ram[0x07f8:0x07fb]))) or '400')

    def _get_reward(self):
        """Return a simple reward based on x-position progress."""
        # Calculate x position reward
        x_position = self.ram[0x6d] * 0x100 + self.ram[0x86]
        x_reward = x_position - self._x_position_last
        self._x_position_last = x_position
        
        # Calculate time penalty
        time = int(''.join(map(str, self.ram[0x07f8:0x07fb])))
        time_penalty = time - self._time_last
        self._time_last = time
        
        # Calculate death penalty
        player_state = self.ram[0x000e]
        death_penalty = -15 if player_state == 0x0b or player_state == 0x06 else 0
        
        return x_reward + time_penalty + death_penalty
    
    def _get_done(self):
        """Return True if the episode is over."""
        # Check if Mario is dead or dying
        player_state = self.ram[0x000e]
        return player_state == 0x0b or player_state == 0x06 or self.ram[0x075a] == 0xff
    
    def _get_info(self):
        """Return information about the current state."""
        return {
            'x_pos': self.ram[0x6d] * 0x100 + self.ram[0x86],
            'y_pos': 255 - self.ram[0x03b8],
            'world': self.ram[0x075f] + 1,
            'stage': self.ram[0x075c] + 1,
            'time': int(''.join(map(str, self.ram[0x07f8:0x07fb]))),
            'coins': int(''.join(map(str, self.ram[0x07ed:0x07ef]))),
            'score': int(''.join(map(str, self.ram[0x07de:0x07e4]))),
            'life': self.ram[0x075a],
            'last_action': self.last_action_info
        }
    
    def reset(self):
        """Reset the environment and return the initial observation with instructions."""
        # Call the parent reset method
        initial_state = super().reset()
        
        # Reset tracking variables
        self._x_position_last = self.ram[0x6d] * 0x100 + self.ram[0x86]
        self._time_last = int(''.join(map(str, self.ram[0x07f8:0x07fb])))
        self.last_action_info = "No actions executed yet."
        
        # Add instructions to the observation
        instructions = """
You are playing Super Mario Bros. Your objective is to advance as far as possible in the game. At each step you can 
submit multiple actions.
Action format: Submit actions in square brackets like [a] or [r].
You can submit multiple actions simultaneously: [l] [a] (jump left)
You can also submit action sequences: [r] + [r] + [r] [a] (move right twice, then jump right)

Available actions:
- [a]: Jump (A button)
- [b]: Run (B button)
- [u]: Move up (for climbing)
- [d]: Move down (for pipes)
- [l]: Move left
- [r]: Move right
- [n]: No operation

Common combinations:
- Jump right: [r] [a]
- Run right: [r] [b]
- Jump while running right: [r] [a] [b]
- Jump left: [l] [a]
- Run left: [l] [b]
        """
        
        # Return observation with instructions
        return {
            'visual': initial_state,
            'text': instructions
        }
    
    def step(self, action_string):
        """
        Process action string, execute actions, and return the result.
        
        Args:
            action_string (str): A string containing actions in [x] format
        
        Returns:
            tuple: (observation, reward, done, info)
        """
        # Extract actions using regex
        action_groups = re.findall(r'\[(.[^\]]*)\]', action_string.lower())
        
        # Process the actions
        if not action_groups:
            self.last_action_info = "No actions were executed because none were provided in the [x] format."
            # Just advance the frame with no action
            state, reward, done, info = super().step(0)
        else:
            total_reward = 0
            executed_actions = []
            
            # Split into time steps based on '+' delimiter
            time_steps = ' '.join(action_groups).split('+')
            
            for step in time_steps:
                # Get all actions at this time step (to be executed simultaneously)
                simultaneous_actions = [a.strip() for a in step.split() if a.strip()]
                
                if simultaneous_actions:
                    # Convert to NES buttons for simultaneous press
                    nes_buttons = []
                    for action_code in simultaneous_actions:
                        if action_code in self.ACTION_MAPPING:
                            nes_buttons.append(self.ACTION_MAPPING[action_code])
                    
                    # Take the step with simultaneous button presses
                    nes_action = self._convert_to_nes_action(nes_buttons)
                    executed_actions.append(f"({'+'.join(simultaneous_actions)})")
                    
                    state, reward, done, info = super().step(nes_action)
                    total_reward += reward
                    
                    # Break if the episode is done
                    if done:
                        break
            
            self.last_action_info = f"Executed actions: {' '.join(executed_actions)}"
            
            # Return the final state with cumulative reward
            reward = total_reward
        
        # Return observation as a dictionary with screen and info
        observation = {
            'screen': state,
            'info': self._get_info()
        }
        
        return observation, reward, done, info
    
    def _convert_to_nes_action(self, action_buttons):
        """
        Convert action buttons to NES-py action bitmask.
        
        Args:
            action_buttons (list): List of button names to press
            
        Returns:
            int: Action bitmask for NES-py
        """
        # NES controller button mapping:
        # 7 6 5 4 3 2 1 0
        # A B S T U D L R
        # (S=Select, T=Start, U=Up, D=Down, L=Left, R=Right)
        
        button_map = {
            'A': 0x80,      # A button
            'B': 0x40,      # B button
            'SELECT': 0x20, # Select button
            'START': 0x10,  # Start button
            'up': 0x08,     # Up
            'down': 0x04,   # Down
            'left': 0x02,   # Left
            'right': 0x01,  # Right
            'NOOP': 0x00,   # No operation
        }
        
        # Start with no buttons pressed
        action = 0x00
        
        # Add each button press to the action
        for button in action_buttons:
            if button in button_map:
                action |= button_map[button]
        
        return action
    
    def render(self, mode='human'):
        """Render the environment."""
        return super().render(mode=mode)

# Helper function to create the environment
def create_mario_env():
    return MinimalMarioEnv()