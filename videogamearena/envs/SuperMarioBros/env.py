
import os, re, time
from nes_py import NESEnv

# class SuperMarioBrosEnv(NESEnv):
#     """A minimal Super Mario Bros environment optimized for language model interaction."""
    
#     # Define the action mapping for language models
#     ACTION_MAPPING = {
#         'a': 'A',           # Jump
#         'b': 'B',           # Run
#         'u': 'up',          # Move up (for climbing)
#         'd': 'down',        # Move down (for pipes)
#         'l': 'left',        # Move left
#         'r': 'right',       # Move right
#         'n': 'NOOP',        # No operation
#     }
    
#     def __init__(self, mode='human'):
#         """
#         Initialize the Mario environment.
        
#         Args:
#             mode (str): Game speed mode - 'slow' or 'human'
#         """
#         # Path to your Super Mario Bros ROM file
#         rom_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'super-mario-bros.nes')
#         super(SuperMarioBrosEnv, self).__init__(rom_path)
        
#         # Initialize variables
#         self._x_position_last = 0
#         self._time_last = 0
#         self.last_action_info = "No actions executed yet."
#         self.mode = mode
#         self.frame_skip = 1 if mode == 'human' else 4
        
#         # Reset the environment
#         self.reset()
#         # Skip the start screen
#         self._skip_start_screen()
#         # Create a backup state
#         self._backup()
    
#     def _skip_start_screen(self):
#         """Press and release start to skip all intro screens and get to actual gameplay."""
#         # Wait a few frames first to make sure the ROM is loaded
#         for _ in range(5):
#             self._frame_advance(0)
        
#         # Press start repeatedly with some delay until we get past the title screen
#         for _ in range(20):
#             # Press start
#             self._frame_advance(8)  # START button (bit 4)
#             # Wait a few frames
#             for _ in range(3):
#                 self._frame_advance(0)
#             # Release start
#             self._frame_advance(0)
            
#             # Check if the game has moved past the title screen
#             if (self.ram[0x075f] > 0 or self.ram[0x075c] > 0):  # World or stage values are set
#                 break
            
#             # Add a small delay between attempts
#             for _ in range(5):
#                 self._frame_advance(0)
        
#         # Now we need to get past the black screen with the world announcement
#         # This screen shows "WORLD 1-1" and "MARIO × 3"
#         # Press start and wait until Mario's x position is initialized
#         for _ in range(30):  # Try multiple times
#             # Press start
#             self._frame_advance(8)
#             # Wait a few frames
#             for _ in range(3):
#                 self._frame_advance(0)
#             # Release start
#             self._frame_advance(0)
            
#             # Check if Mario's position has been initialized to something other than zero
#             # When actual gameplay starts, Mario will have a non-zero x position
#             if self.ram[0x86] > 20:  # Mario has a starting position (typically around 40)
#                 break
                
#             # Add a small delay between attempts
#             for _ in range(5):
#                 self._frame_advance(0)
        
#         # Final waiting period to ensure the game is fully loaded
#         for _ in range(10):
#             self._frame_advance(0)
        
#         # Update the position and time trackers
#         self._x_position_last = self.ram[0x6d] * 0x100 + self.ram[0x86]
#         self._time_last = int(''.join(map(str, filter(lambda x: x != 0, self.ram[0x07f8:0x07fb]))) or '400')
    
#     def _get_reward(self):
#         """Return a simple reward based on x-position progress."""
#         # Calculate x position reward
#         x_position = self.ram[0x6d] * 0x100 + self.ram[0x86]
#         x_reward = x_position - self._x_position_last
#         self._x_position_last = x_position
        
#         # Calculate time penalty
#         time = int(''.join(map(str, filter(lambda x: x != 0, self.ram[0x07f8:0x07fb]))) or '0')
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
#             'time': int(''.join(map(str, filter(lambda x: x != 0, self.ram[0x07f8:0x07fb]))) or '0'),
#             'coins': int(''.join(map(str, filter(lambda x: x != 0, self.ram[0x07ed:0x07ef]))) or '0'),
#             'score': int(''.join(map(str, filter(lambda x: x != 0, self.ram[0x07de:0x07e4]))) or '0'),
#             'life': self.ram[0x075a],
#             'last_action': self.last_action_info
#         }
    
#     def reset(self):
#         """Reset the environment and return the initial observation with instructions."""
#         # Call the parent reset method
#         initial_state = super().reset()
        
#         # Reset tracking variables
#         self._x_position_last = self.ram[0x6d] * 0x100 + self.ram[0x86]
#         self._time_last = int(''.join(map(str, filter(lambda x: x != 0, self.ram[0x07f8:0x07fb]))) or '400')
#         self.last_action_info = "No actions executed yet."
        
#         # Add instructions to the observation
#         instructions = """
# Action format: Submit actions in square brackets like [a] or [r].
# You can submit multiple actions simultaneously: [l] [a] (jump left)
# You can also submit action sequences: [r] + [r] + [r] [a] (move right twice, then jump right)

# Available actions:
# - [a]: Jump (A button)
# - [b]: Run (B button)
# - [u]: Move up (for climbing)
# - [d]: Move down (for pipes)
# - [l]: Move left
# - [r]: Move right
# - [n]: No operation

# Common combinations:
# - Jump right: [r] [a]
# - Run right: [r] [b]
# - Jump while running right: [r] [a] [b]
# - Jump left: [l] [a]
# - Run left: [l] [b]
#         """
        
#         # Return observation with visuals and text
#         return {
#             'visual': initial_state,
#             'text': instructions
#         }
    
#     def step(self, action_string):
#         """
#         Process action string, execute actions, and return the result.
#         The game will continue running even without explicit actions.
        
#         Args:
#             action_string (str): A string containing actions in [x] format
        
#         Returns:
#             tuple: (observation, reward, done, info)
#         """
#         # Extract actions using regex
#         action_groups = re.findall(r'\[(.[^\]]*)\]', action_string.lower())
        
#         # Process the actions
#         if not action_groups:
#             self.last_action_info = "No actions were executed because none were provided in the [x] format."
#             # Just advance the frame with no action (game continues running)
#             state, reward, done, info = self._step_game_with_no_action()
#         else:
#             total_reward = 0
#             executed_actions = []
            
#             # Split into time steps based on '+' delimiter
#             time_steps = ' '.join(action_groups).split('+')
            
#             for step in time_steps:
#                 # Get all actions at this time step (to be executed simultaneously)
#                 simultaneous_actions = [a.strip() for a in step.split() if a.strip()]
                
#                 if simultaneous_actions:
#                     # Convert to NES buttons for simultaneous press
#                     nes_buttons = []
#                     for action_code in simultaneous_actions:
#                         if action_code in self.ACTION_MAPPING:
#                             nes_buttons.append(self.ACTION_MAPPING[action_code])
                    
#                     # Take the step with simultaneous button presses
#                     nes_action = self._convert_to_nes_action(nes_buttons)
#                     executed_actions.append(f"({'+'.join(simultaneous_actions)})")
                    
#                     # Step the game with the action and apply frame skip based on mode
#                     state, reward, done, info = self._step_game_with_action(nes_action)
#                     total_reward += reward
                    
#                     # Break if the episode is done
#                     if done:
#                         break
            
#             self.last_action_info = f"Executed actions: {' '.join(executed_actions)}"
#             reward = total_reward
        
#         # Return observation as a dictionary with visuals and text information
#         observation = {
#             'visual': state,
#             'text': self._get_formatted_info_text()
#         }
        
#         return observation, reward, done, info
    
#     def _step_game_with_no_action(self):
#         """Step the game with no action (NOOP)."""
#         reward_sum = 0
#         done = False
#         info = None
        
#         # Apply frame skip based on mode
#         for _ in range(self.frame_skip):
#             state, reward, done, info = super().step(0)  # NOOP action
#             reward_sum += reward
#             if done:
#                 break
        
#         return state, reward_sum, done, info
    
#     def _step_game_with_action(self, action):
#         """Step the game with a specific action."""
#         reward_sum = 0
#         done = False
#         info = None
        
#         # Apply frame skip based on mode
#         for _ in range(self.frame_skip):
#             state, reward, done, info = super().step(action)
#             reward_sum += reward
#             if done:
#                 break
        
#         return state, reward_sum, done, info
    
#     def _convert_to_nes_action(self, action_buttons):
#         """
#         Convert action buttons to NES-py action bitmask.
        
#         Args:
#             action_buttons (list): List of button names to press
            
#         Returns:
#             int: Action bitmask for NES-py
#         """
#         # NES controller button mapping:
#         # 7 6 5 4 3 2 1 0
#         # A B S T U D L R
#         # (S=Select, T=Start, U=Up, D=Down, L=Left, R=Right)
        
#         button_map = {
#             'A': 0x80,      # A button
#             'B': 0x40,      # B button
#             'SELECT': 0x20, # Select button
#             'START': 0x10,  # Start button
#             'up': 0x08,     # Up
#             'down': 0x04,   # Down
#             'left': 0x02,   # Left
#             'right': 0x01,  # Right
#             'NOOP': 0x00,   # No operation
#         }
        
#         # Start with no buttons pressed
#         action = 0x00
        
#         # Add each button press to the action
#         for button in action_buttons:
#             if button in button_map:
#                 action |= button_map[button]
        
#         return action
    
#     def _get_formatted_info_text(self):
#         """Format the game information into a readable text string."""
#         info = self._get_info()
#         text = f"""
# World: {info['world']}-{info['stage']}
# Position: ({info['x_pos']}, {info['y_pos']})
# Coins: {info['coins']} | Lives: {info['life']} | Time: {info['time']}
# {info['last_action']}
# """
#         return text
    
#     def render(self, mode='human'):
#         """Render the environment."""
#         return super().render(mode=mode)
    
#     def set_mode(self, mode):
#         """Set the game speed mode."""
#         if mode not in ['slow', 'human']:
#             raise ValueError("Mode must be 'slow' or 'human'")
#         self.mode = mode
#         self.frame_skip = 1 if mode == 'human' else 4

class SuperMarioBrosEnv(NESEnv):
    """A minimal Super Mario Bros environment optimized for language model interaction."""
    
    # Define the action mapping
    ACTION_MAPPING = {
        'a': 'A',           # Jump
        'b': 'B',           # Run
        'u': 'up',          # Move up (for climbing)
        'd': 'down',        # Move down (for pipes)
        'l': 'left',        # Move left
        'r': 'right',       # Move right
        'n': 'NOOP',        # No operation
    }
    
    def __init__(self, mode='human'):
        """
        Initialize the Mario environment.
        
        Args:
            mode (str): Game speed mode - 'slow' or 'human'
        """
        # Path to your Super Mario Bros ROM file
        rom_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'super-mario-bros.nes')
        super(SuperMarioBrosEnv, self).__init__(rom_path)
        
        # Initialize variables
        self._x_position_last = 0
        self._time_last = 0
        self.last_action_info = "No actions executed yet."
        self.mode = mode
        self.frame_skip = 1 if mode == 'human' else 4
        self.total_reward = 0
        
        # Reset the environment
        self.reset()
        # Skip the start screen
        self._skip_start_screen()
        # Create a backup state
        self._backup()
    
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
        time = int(''.join(map(str, filter(lambda x: x != 0, self.ram[0x07f8:0x07fb]))) or '0')
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
            'time': int(''.join(map(str, filter(lambda x: x != 0, self.ram[0x07f8:0x07fb]))) or '0'),
            'coins': int(''.join(map(str, filter(lambda x: x != 0, self.ram[0x07ed:0x07ef]))) or '0'),
            'score': int(''.join(map(str, filter(lambda x: x != 0, self.ram[0x07de:0x07e4]))) or '0'),
            'life': self.ram[0x075a],
            'last_action': self.last_action_info
        }
    
    def reset(self):
        """Reset the environment and return the initial observation with instructions."""
        # Call the parent reset method
        initial_state = super().reset()
        
        # Reset tracking variables
        self._x_position_last = self.ram[0x6d] * 0x100 + self.ram[0x86]
        self._time_last = int(''.join(map(str, filter(lambda x: x != 0, self.ram[0x07f8:0x07fb]))) or '400')
        self.last_action_info = "No actions executed yet."
        self.total_reward = 0
        
        # Add instructions to the observation
        instructions = """
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
        
        # Return observation with visuals and text
        return {
            'visual': initial_state,
            'text': instructions
        }
    
    def step(self, action_string):
        """
        Process action string, execute actions, and return the result.
        The game will continue running even without explicit actions.
        
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
            # Just advance the frame with no action (game continues running)
            state, reward, done, info = self._step_game_with_no_action()
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
                    
                    # Step the game with the action and apply frame skip based on mode
                    state, reward, done, info = self._step_game_with_action(nes_action)
                    total_reward += reward
                    
                    # Break if the episode is done
                    if done:
                        break
            
            self.last_action_info = f"Executed actions: {' '.join(executed_actions)}"
            reward = total_reward
        
        # Update total reward
        self.total_reward += reward
        
        # Return observation as a dictionary with visuals and text information
        observation = {
            'visual': state,
            'text': self._get_formatted_info_text()
        }
        
        return observation, reward, done, info
    
    def _step_game_with_no_action(self):
        """Step the game with no action (NOOP)."""
        reward_sum = 0
        done = False
        info = None
        
        # Apply frame skip based on mode
        for _ in range(self.frame_skip):
            state, reward, done, info = super().step(0)  # NOOP action
            reward_sum += reward
            if done:
                break
        time.sleep(0.05)
        return state, reward_sum, done, info
    
    def _step_game_with_action(self, action):
        """Step the game with a specific action."""
        reward_sum = 0
        done = False
        info = None
        
        # Apply frame skip based on mode
        for _ in range(self.frame_skip):
            state, reward, done, info = super().step(action)
            reward_sum += reward
            if done:
                break
        
        time.sleep(0.05)
        return state, reward_sum, done, info
    
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
    
    def _get_formatted_info_text(self):
        """Format the game information into a readable text string."""
        info = self._get_info()
        text = f"""
World: {info['world']}-{info['stage']}
Position: ({info['x_pos']}, {info['y_pos']})
Coins: {info['coins']} | Lives: {info['life']} | Time: {info['time']}
Score: {info['score']} | Total Reward: {self.total_reward:.2f}
{info['last_action']}
"""
        return text
    
    def close(self):
        """Close the environment and return the total reward."""
        super().close()
        return self.total_reward
    
    def set_mode(self, mode):
        """Set the game speed mode."""
        if mode not in ['slow', 'human']:
            raise ValueError("Mode must be 'slow' or 'human'")
        self.mode = mode
        self.frame_skip = 1 if mode == 'human' else 4