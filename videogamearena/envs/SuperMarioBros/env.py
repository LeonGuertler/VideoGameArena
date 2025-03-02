from nes_py.wrappers import JoypadSpace
from nes_py import NESEnv
import os, re, time
import numpy as np
from typing import Union, Dict



# _button_map = {
#     'right':  0b10000000,
#     'left':   0b01000000,
#     'down':   0b00100000,
#     'up':     0b00010000,
#     'start':  0b00001000,
#     'select': 0b00000100,
#     'B':      0b00000010,
#     'A':      0b00000001,
#     'NOOP':   0b00000000,
# }



class SuperMarioBrosEnv:
    """A simplified Super Mario Bros environment that uses JoypadSpace for actions."""
    
    # Define the action mapping for the JoypadSpace wrapper
    ACTIONS = [
        ['NOOP'],
        ['right'],
        ['right', 'A'],
        ['right', 'B'],
        ['right', 'A', 'B'],
        ['A'],
        ['left'],
        ['left', 'A'],
        ['left', 'B'],
        ['left', 'A', 'B'],
        ['down'],
        ['up'],
    ]
    
    # Define mapping from our simple action codes to JoypadSpace action indices
    ACTION_MAPPING = {
        "u": 0b00010000,
        "d": 0b00100000,
        "l": 0b01000000,
        "r": 0b10000000,
        "a": 0b00000001,
        "b": 0b00000010,
        "o": 0b00001000, # start
        "p": 0b00000100, # select
        "n": 0b00000000,
    }




    #     'n': 0,         # NOOP
    #     'r': 1,         # right
    #     'ra': 2,        # right + A (jump right)
    #     'rb': 3,        # right + B (run right)
    #     'rab': 4,       # right + A + B (jump while running right)
    #     'a': 5,         # A (jump)
    #     'l': 6,         # left
    #     'la': 7,        # left + A (jump left)
    #     'lb': 8,        # left + B (run left)
    #     'lab': 9,       # left + A + B (jump while running left)
    #     'd': 10,        # down
    #     'u': 11,        # up
    # }
    
    def __init__(self, mode='human'):
        """
        Initialize the Mario environment.
        
        Args:
            mode (str): Game speed mode - 'slow' or 'human'
        """
        # Path to your Super Mario Bros ROM file
        rom_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'super-mario-bros.nes')
        
        # Create the base NES environment
        self.env = NESEnv(rom_path)
        
        # Wrap it with JoypadSpace to use simplified actions
        # self.env = JoypadSpace(self.base_env, self.ACTIONS)
        
        # Initialize variables
        self._x_position_last = 0
        self._time_last = 0
        self.last_action_info = "No actions executed yet."
        self.mode = mode
        self.frame_skip = 1 if mode == 'human' else 4
        self.total_reward = 0
        
        # Simple reset without trying to skip start screen
        self.env.reset()
        
        # Note: We'll let the user or model press start button directly
    
    def _initialize_game(self):
        """Complete initialization process including bypassing start screens."""
        print("Initializing Mario environment...")
        
        # First, reset the environment to get to a stable state
        initial_state = self.env.reset()
        
        # Now we'll try to bypass the title screen
        START_BUTTON = 8  # START is bit 3 (value 8)
        
        print("Attempting to bypass title screen...")
        # Press START repeatedly with longer intervals
        for attempt in range(10):
            print(f"Start attempt {attempt+1}")
            
            # Hold START for several frames
            for _ in range(5):
                # Use the base environment to ensure direct control
                self.env.step(START_BUTTON)
            
            # Release all buttons for several frames
            for _ in range(10):
                self.env.step(0)
            
            # Check if game has started by looking at the timer
            if self._get_time() > 0:
                print("Game appears to have started!")
                break
        
        # Give the game more time to fully initialize
        for _ in range(30):
            self.env.step(0)
        
        # Final check and initialization
        self._time_last = self._get_time()
        self._x_position_last = self._get_x_position()
        
        print(f"Game state: World {self._get_world()}-{self._get_stage()}, Time: {self._get_time()}")
        
        if self._get_time() > 0:
            print("Game successfully initialized!")
        else:
            print("WARNING: Game initialization may have failed.")
    
    def _get_world(self):
        """Get current world number."""
        return self.env.unwrapped.ram[0x075f] + 1
    
    def _get_stage(self):
        """Get current stage/level number."""
        return self.env.unwrapped.ram[0x075c] + 1

    def _get_time(self):
        """Get the current game time."""
        time_digits = self.env.unwrapped.ram[0x07f8:0x07fb]
        return int(''.join(map(str, filter(lambda x: x != 0, time_digits))) or '400')

    def reset(self):
        """Reset the environment and return the initial observation with instructions."""
        # Call the underlying reset method
        initial_state = self.env.reset()
        
        # Reset tracking variables
        self._x_position_last = self.env.unwrapped.ram[0x6d] * 0x100 + self.env.unwrapped.ram[0x86]
        self._time_last = self._get_time()
        self.last_action_info = "No actions executed yet."
        self.total_reward = 0
        
        # Add instructions to the observation
        instructions = """
Action format: Submit actions in square brackets like [a] or [r].
You can submit multiple actions simultaneously: [r] [a] (equivalent to [ra])
You can also submit action sequences: [r] + [r] + [ra] (move right twice, then jump right)

Available actions:
- [a]: Jump (A button)
- [b]: Run (B button)
- [u]: Move up (for climbing)
- [d]: Move down (for pipes)
- [l]: Move left
- [r]: Move right
- [n]: No operation

Combinations (you can also use these directly):
- [ra]: Jump right (right + A)
- [rb]: Run right (right + B)
- [rab]: Jump while running right (right + A + B)
- [la]: Jump left (left + A)
- [lb]: Run left (left + B)
- [lab]: Jump while running left (left + A + B)
        """
        
        # Return observation with visuals and text
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
        
        # Special handling for start button (not in our regular action space)
        if '[start]' in action_string.lower():
            print("Pressing START button...")
            # START button is bit 3 (value 8) in NES controller
            self.env.step(8)  # Press START
            # Get the screen observation directly without using private methods
            state = self.env.render(mode='rgb_array')
            self.env.step(0)  # Release all buttons
            
            reward = 0
            done = False
            info = {}
            self.last_action_info = "Executed START button press"
        # Process the regular actions
        elif not action_groups:
            self.last_action_info = "No actions were executed because none were provided in the [x] format."
            # Just advance the frame with no action (game continues running)
            state, reward, done, info = self._step_with_action(0)  # NOOP
        else:
            total_reward = 0
            executed_actions = []

            # execute all submitted actions in the env
            # Step the environment with the action

            legal_bytes = [
                0b10000001, # ('r', 'a'),
                0b10000010, # ('r', 'b'),
                0b10000011, # ('r', 'a', 'b'),
                0b01000001, # ('l', 'a'),
                0b01000010, # ('l', 'b'),
                0b01000011, # ('l', 'a', 'b'),
                0b00010000,
                0b00100000,
                0b01000000,
                0b10000000,
                0b00000001,
                0b00000010,
                0b00001000, # start
                0b00000100, # select
                0b00000000,
            ]
            # legal_combinations = [
            #     ('r', 'a'),
            #     ('r', 'b'),
            #     ('r', 'a', 'b'),
            #     ('l', 'a'),
            #     ('l', 'b'),
            #     ('l', 'a', 'b'),
            # ]
            action = 0b00000000
            for a in action_groups:
                action += self.ACTION_MAPPING[a]
            
            if action in legal_bytes:
                state, reward, done, info = self._step_with_action(action)
                total_reward += reward
            else:
                state, reward, done, info = self._step_with_action(0)  # NOOP
                print(f"Not a legal combination: {bin(action)}")


            # ACTION_MAPPING = {
            #     "u": 0b00010000,
            #     "d": 0b00100000,
            #     "l": 0b01000000,
            #     "r": 0b10000000,
            #     "a": 0b00000001,
            #     "b": 0b00000010,
            #     "o": 0b00001000, # start
            #     "p": 0b00000100, # select
            #     "n": 0b00000000,
            # }



            # # convert action groups to set
            # if tuple(action_groups) in legal_combinations:
            #     # sum up the bytes
            #     action = 0b00000000

            #     for a in action_groups:


            # state, reward, done, info = self._step_with_action([self.ACTION_MAPPING[a] for a in action_groups])
            # for action in action_groups:
            #     state, reward, done, info = self._step_with_action(self.ACTION_MAPPING[action])

            
            # Print debug info

            # print(f"Executing: {executed_actions[-1]} (action index: {action_idx})")
                    
            
            # # Split into time steps based on '+' delimiter
            # time_steps = ' '.join(action_groups).split('+')
            
            # for step in time_steps:
            #     # Get all actions at this time step (to be executed simultaneously)
            #     simultaneous_actions = [a.strip() for a in step.split() if a.strip()]
                
            #     if simultaneous_actions:
            #         # Check if the combination is directly in our mapping
            #         if len(simultaneous_actions) == 1 and simultaneous_actions[0] in self.ACTION_MAPPING:
            #             # Single action that maps directly
            #             action_idx = self.ACTION_MAPPING[simultaneous_actions[0]]
            #             executed_actions.append(f"{simultaneous_actions[0]}")
            #         else:
            #             # Process multiple actions to determine the combined action

            #             combined_action = self._combine_actions(simultaneous_actions)
            #             if combined_action in self.ACTION_MAPPING:
            #                 action_idx = self.ACTION_MAPPING[combined_action]
            #                 executed_actions.append(f"{combined_action}")
            #             else:
            #                 # If the combination isn't valid, use NOOP
            #                 action_idx = 0
            #                 executed_actions.append("n (invalid combination)")
                    
            #         # Step the environment with the action
            #         state, reward, done, info = self._step_with_action(action_idx)
            #         total_reward += reward
                    
            #         # Print debug info
            #         print(f"Executing: {executed_actions[-1]} (action index: {action_idx})")
                    
            #         # Break if the episode is done
            #         if done:
            #             break
            
            self.last_action_info = f"Executed actions: {' '.join(executed_actions)}"
            reward = total_reward
        
        # Return observation as a dictionary with visuals and text information
        observation = {
            'visual': state,
            'text': self._get_formatted_info_text()
        }
        time.sleep(0.03)
        return observation, reward, done, info
    
    def _combine_actions(self, actions):
        """
        Combine multiple action codes into a single composite action.
        
        Args:
            actions (list): List of action codes like ['r', 'a']
            
        Returns:
            str: Combined action code like 'ra'
        """
        # Special case handling - if conflicting directions, use the last one
        directions = {'u', 'd', 'l', 'r'}
        dir_actions = [a for a in actions if a in directions]
        if len(dir_actions) > 1:
            # Only keep the last direction
            for d in dir_actions[:-1]:
                actions.remove(d)
        
        # Combine the remaining actions
        return ''.join(sorted(actions))
    
    def _step_with_action(self, action_idx):
        """
        Step the environment with the given action index.
        
        Args:
            action_idx (int): Index into the ACTIONS list
            
        Returns:
            tuple: (state, reward, done, info)
        """
        reward_sum = 0
        done = False
        info = None
        
        # Apply frame skip based on mode
        for _ in range(self.frame_skip):
            state, reward, done, info = self.env.step(action_idx)
            reward_sum += reward
            if done:
                break
        
        # Update total reward
        self.total_reward += reward_sum
        
        return state, reward_sum, done, info
    
    def _get_formatted_info_text(self):
        """Format the game information into a readable text string."""
        # Get information from the underlying environment
        ram = self.env.unwrapped.ram
        x_pos = self._get_x_position()
        y_pos = 255 - ram[0x03b8]
        world = ram[0x075f] + 1
        stage = ram[0x075c] + 1
        time_digits = ram[0x07f8:0x07fb]
        try:
            time = int(''.join(map(str, filter(lambda x: x != 0, time_digits))) or '0')
        except:
            time = 0
        coin_digits = ram[0x07ed:0x07ef]
        try:
            coins = int(''.join(map(str, filter(lambda x: x != 0, coin_digits))) or '0')
        except:
            coins = 0
        life = ram[0x075a]
        
        text = f"""
World: {world}-{stage}
Position: ({x_pos}, {y_pos})
Coins: {coins} | Lives: {life} | Time: {time}
Total Reward: {self.total_reward:.2f}
{self.last_action_info}
"""
        return text
        
    def _get_x_position(self):
        """Get Mario's x position in the level."""
        ram = self.env.unwrapped.ram
        return ram[0x6d] * 0x100 + ram[0x86]
    
    def render(self, mode='human'):
        """Render the environment."""
        return self.env.render(mode=mode)
    
    def close(self):
        """Close the environment and return the total reward."""
        self.env.close()
        return self.total_reward
    
    def set_mode(self, mode):
        """Set the game speed mode."""
        if mode not in ['slow', 'human']:
            raise ValueError("Mode must be 'slow' or 'human'")
        self.mode = mode
        self.frame_skip = 1 if mode == 'human' else 4