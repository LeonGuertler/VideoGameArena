import os, re, time
import numpy as np
from nes_py import NESEnv

from typing import Union, Dict


class SuperMarioBrosEnv:
    """ TODO """

    BYTE_MAPPING = {
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

    LEGAL_BYTES = [
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

    # Frame rate constants for different speed modes
    FPS = {
        'human': 30,        # 30 fps - normal NES speed
        'slow': 15,         # 15 fps - half speed
        'super-slow': 5     # 5 fps - very slow for detailed analysis
    }

    def __init__(self, mode='slow'):
        """
        Initialize the Mario environment.
        
        Args:
            mode (str): Game speed mode - 'human', 'slow', or 'super-slow'
        """
        if mode not in self.FPS:
            raise ValueError(f"Mode must be one of {list(self.FPS.keys())}")
            
        # Path to your Super Mario Bros ROM file
        rom_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'super-mario-bros.nes')
        
        # Create the base NES environment
        self.env = NESEnv(rom_path)
        
        # Initialize variables
        self._x_position_last = 0
        self._time_last = 0
        self.last_action_info = "No actions executed yet."
        self.mode = mode
        # Use frame_skip=4 consistently across all modes
        # self.frame_skip = 4
        self.total_reward = 0
        
        # Set target frame time based on mode
        self.target_frame_time = 1.0 / self.FPS[mode]
        self.last_frame_time = time.time()
        
        # Simple reset without trying to skip start screen
        self.env.reset()
        
    
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
        # Reset the frame timer
        self.last_frame_time = time.time()
        
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
        # Start timing this frame
        frame_start_time = time.time()
        
        # Extract actions using regex
        action_groups = re.findall(r'\[(.[^\]]*)\]', action_string.lower())
        

        # Process the regular actions
        if not action_groups:
            self.last_action_info = "No actions were executed because none were provided in the [x] format."
            # Just advance the frame with no action (game continues running)
            state, reward, done, info = self._step_with_action(0)  # NOOP
        else:
            total_reward = 0
            executed_actions = []

            action = 0b00000000
            for a in action_groups:
                action += self.BYTE_MAPPING[a]
            
            if action in self.LEGAL_BYTES:
                state, reward, done, info = self._step_with_action(action)
                total_reward += reward
                executed_actions.append(''.join(action_groups))
            else:
                state, reward, done, info = self._step_with_action(0)  # NOOP
                print(f"Not a legal combination: {bin(action)}")

            self.last_action_info = f"Executed actions: {' '.join(executed_actions)}"
            reward = total_reward
        
        # Return observation as a dictionary with visuals and text information
        observation = {
            'visual': state,
            'text': self._get_formatted_info_text()
        }
        
        # Calculate how long this frame took to process
        frame_execution_time = time.time() - frame_start_time
        
        # Calculate the time to sleep to maintain the target FPS
        sleep_time = max(0, self.target_frame_time - frame_execution_time)
        
        # Add debug info about frame timing
        print(f"Target FPS: {self.FPS[self.mode]}, Frame time: {frame_execution_time*1000:.1f}ms, Sleep: {sleep_time*1000:.1f}ms")
        if 'text' in observation:
            fps_info = f"Target FPS: {self.FPS[self.mode]}, Frame time: {frame_execution_time*1000:.1f}ms, Sleep: {sleep_time*1000:.1f}ms"
            observation['text'] += f"\n{fps_info}"
        
        # Sleep the appropriate amount to maintain constant frame rate
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        # Update the last frame time
        self.last_frame_time = time.time()
        
        return observation, reward, done, info
    

    def _step_with_action(self, action_byte):
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

        state, reward, done, info = self.env.step(action_byte)
        
        # Update total reward
        self.total_reward += reward
        
        return state, reward, done, info
    
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
Mode: {self.mode} ({self.FPS[self.mode]} FPS)
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
        if mode not in self.FPS:
            raise ValueError(f"Mode must be one of {list(self.FPS.keys())}")
        self.mode = mode
        # Always use frame_skip=4 across all modes
        self.frame_skip = 4
        self.target_frame_time = 1.0 / self.FPS[mode]
        print(f"Game speed set to {mode} mode ({self.FPS[mode]} FPS, frame_skip=4)")