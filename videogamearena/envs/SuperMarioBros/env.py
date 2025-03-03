# strongly based on this amazing work: https://github.com/Kautenja/gym-super-mario-bros/blob/master/gym_super_mario_bros/smb_env.py#L282

import os, re, time
from collections import defaultdict
from nes_py import NESEnv
import numpy as np



# create a dictionary mapping value of status register to string names
_STATUS_MAP = defaultdict(lambda: 'fireball', {0:'small', 1: 'tall'})


# a set of state values indicating that Mario is "busy"
_BUSY_STATES = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x07]


# RAM addresses for enemy types on the screen
_ENEMY_TYPE_ADDRESSES = [0x0016, 0x0017, 0x0018, 0x0019, 0x001A]


# enemies whose context indicate that a stage change will occur (opposed to an
# enemy that implies a stage change wont occur -- i.e., a vine)
# Bowser = 0x2D
# Flagpole = 0x31
_STAGE_OVER_ENEMIES = np.array([0x2D, 0x31])


class SuperMarioBrosEnv(NESEnv):
    """An environment for playing Super Mario Bros with OpenAI Gym and frame rate control."""

    # the legal range of rewards for each step
    reward_range = (-15, 15)
    
    # Byte mapping for input actions
    BYTE_MAPPING = {
        "u": 0b00010000,  # Up
        "d": 0b00100000,  # Down
        "l": 0b01000000,  # Left
        "r": 0b10000000,  # Right
        "a": 0b00000001,  # A button
        "b": 0b00000010,  # B button
        "o": 0b00001000,  # start
        "p": 0b00000100,  # select
        "n": 0b00000000,  # no operation
    }

    # Legal input bytes (combinations allowed)
    LEGAL_BYTES = [
        0b10000001,  # ('r', 'a'),
        0b10000010,  # ('r', 'b'),
        0b10000011,  # ('r', 'a', 'b'),
        0b01000001,  # ('l', 'a'),
        0b01000010,  # ('l', 'b'),
        0b01000011,  # ('l', 'a', 'b'),
        0b00010000,  # up
        0b00100000,  # down
        0b01000000,  # left
        0b10000000,  # right
        0b00000001,  # A
        0b00000010,  # B
        0b00001000,  # start
        0b00000100,  # select
        0b00000000,  # no-op
    ]

    # Frame rate constants for different speed modes
    FPS = {
        'human': 30,       # 30 fps - normal NES speed
        'slow': 15,        # 15 fps - half speed
        'super-slow': 5    # 5 fps - very slow for detailed analysis
    }

    def __init__(self, speed_mode='human'):
        """
        Initialize a new Super Mario Bros environment.

        Args:
            lost_levels (bool): whether to load the ROM with lost levels.
                - False: load original Super Mario Bros.
                - True: load Super Mario Bros. Lost Levels
            target (tuple): a tuple of the (world, stage) to play as a level
            speed_mode (str): Game speed mode - 'human', 'slow', or 'super-slow'

        Returns:
            None
        """
        # Check if the speed mode is valid
        if speed_mode not in self.FPS:
            raise ValueError(f"Speed mode must be one of {list(self.FPS.keys())}")
        
        # decode the ROM path based on mode and lost levels flag
        rom = "videogamearena/envs/SuperMarioBros/super-mario-bros.nes" #rom_path(lost_levels, rom_mode)
        # initialize the super object with the ROM path
        super(SuperMarioBrosEnv, self).__init__(rom)
        # set the target world, stage, and area variables
        self._target_world = None 
        self._target_stage = None 
        self._target_area = None
        # setup a variable to keep track of the last frames time
        self._time_last = 0
        # setup a variable to keep track of the last frames x position
        self._x_position_last = 0
        
        # Setup frame rate control variables
        self.speed_mode = speed_mode
        self.target_frame_time = 1.0 / self.FPS[speed_mode]
        self.last_frame_time = time.time()

        # Setup UI tracking variables
        self.last_action_info = "No actions executed yet."
        self.total_reward = 0
        
        # reset the emulator
        self.reset()
        # skip the start screen
        self._skip_start_screen()
        # create a backup state to restore from on subsequent calls to reset
        self._backup()

    # MARK: Frame rate control methods
    
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

    # MARK: Action parsing methods
    
    def parse_action_string(self, action_string):
        """
        Parse an action string into a byte action.
        
        Args:
            action_string (str): A string containing actions in [x] format
            
        Returns:
            int: The action byte
        """
        # Extract actions using regex
        action_groups = re.findall(r'\[(.[^\]]*)\]', action_string.lower())
        
        if not action_groups:
            self.last_action_info = "No actions were executed because none were provided in the [x] format."
            return 0  # NOOP
        
        # Combine actions
        action = 0b00000000
        for a in action_groups:
            for char in a:
                if char in self.BYTE_MAPPING:
                    action |= self.BYTE_MAPPING[char]
        
        if action not in self.LEGAL_BYTES:
            print(f"Not a legal combination: {bin(action)}")
            action = 0  # Default to NOOP if illegal
        
        self.last_action_info = f"Executed action: {''.join(action_groups)}"
        return action

    # MARK: Memory access

    def _read_mem_range(self, address, length):
        """
        Read a range of bytes where each byte is a 10's place figure.

        Args:
            address (int): the address to read from as a 16 bit integer
            length: the number of sequential bytes to read

        Note:
            this method is specific to Mario where three GUI values are stored
            in independent memory slots to save processing time
            - score has 6 10's places
            - coins has 2 10's places
            - time has 3 10's places

        Returns:
            the integer value of this 10's place representation

        """
        return int(''.join(map(str, self.ram[address:address + length])))

    @property
    def _level(self):
        """Return the level of the game."""
        return self.ram[0x075f] * 4 + self.ram[0x075c]

    @property
    def _world(self):
        """Return the current world (1 to 8)."""
        return self.ram[0x075f] + 1

    @property
    def _stage(self):
        """Return the current stage (1 to 4)."""
        return self.ram[0x075c] + 1

    @property
    def _area(self):
        """Return the current area number (1 to 5)."""
        return self.ram[0x0760] + 1

    @property
    def _score(self):
        """Return the current player score (0 to 999990)."""
        # score is represented as a figure with 6 10's places
        return self._read_mem_range(0x07de, 6)

    @property
    def _time(self):
        """Return the time left (0 to 999)."""
        # time is represented as a figure with 3 10's places
        try:
            return self._read_mem_range(0x07f8, 3)
        except:
            return 0

    @property
    def _coins(self):
        """Return the number of coins collected (0 to 99)."""
        # coins are represented as a figure with 2 10's places
        return self._read_mem_range(0x07ed, 2)

    @property
    def _life(self):
        """Return the number of remaining lives."""
        return self.ram[0x075a]

    @property
    def _x_position(self):
        """Return the current horizontal position."""
        # add the current page 0x6d to the current x
        return self.ram[0x6d] * 0x100 + self.ram[0x86]

    @property
    def _left_x_position(self):
        """Return the number of pixels from the left of the screen."""
        # TODO: resolve RuntimeWarning: overflow encountered in ubyte_scalars
        # subtract the left x position 0x071c from the current x 0x86
        # return (self.ram[0x86] - self.ram[0x071c]) % 256
        return np.uint8(int(self.ram[0x86]) - int(self.ram[0x071c])) % 256

    @property
    def _y_pixel(self):
        """Return the current vertical position."""
        return self.ram[0x03b8]

    @property
    def _y_viewport(self):
        """
        Return the current y viewport.

        Note:
            1 = in visible viewport
            0 = above viewport
            > 1 below viewport (i.e. dead, falling down a hole)
            up to 5 indicates falling into a hole

        """
        return self.ram[0x00b5]

    @property
    def _y_position(self):
        """Return the current vertical position."""
        # check if Mario is above the viewport (the score board area)
        if self._y_viewport < 1:
            # y position overflows so we start from 255 and add the offset
            return 255 + (255 - self._y_pixel)
        # invert the y pixel into the distance from the bottom of the screen
        return 255 - self._y_pixel

    @property
    def _player_status(self):
        """Return the player status as a string."""
        return _STATUS_MAP[self.ram[0x0756]]

    @property
    def _player_state(self):
        """
        Return the current player state.

        Note:
            0x00 : Leftmost of screen
            0x01 : Climbing vine
            0x02 : Entering reversed-L pipe
            0x03 : Going down a pipe
            0x04 : Auto-walk
            0x05 : Auto-walk
            0x06 : Dead
            0x07 : Entering area
            0x08 : Normal
            0x09 : Cannot move
            0x0B : Dying
            0x0C : Palette cycling, can't move

        """
        return self.ram[0x000e]

    @property
    def _is_dying(self):
        """Return True if Mario is in dying animation, False otherwise."""
        return self._player_state == 0x0b or self._y_viewport > 1

    @property
    def _is_dead(self):
        """Return True if Mario is dead, False otherwise."""
        return self._player_state == 0x06

    @property
    def _is_game_over(self):
        """Return True if the game has ended, False otherwise."""
        # the life counter will get set to 255 (0xff) when there are no lives
        # left. It goes 2, 1, 0 for the 3 lives of the game
        return self._life == 0xff

    @property
    def _is_busy(self):
        """Return boolean whether Mario is busy with in-game garbage."""
        return self._player_state in _BUSY_STATES

    @property
    def _is_world_over(self):
        """Return a boolean determining if the world is over."""
        # 0x0770 contains GamePlay mode:
        # 0 => Demo
        # 1 => Standard
        # 2 => End of world
        return self.ram[0x0770] == 2

    @property
    def _is_stage_over(self):
        """Return a boolean determining if the level is over."""
        # iterate over the memory addresses that hold enemy types
        for address in _ENEMY_TYPE_ADDRESSES:
            # check if the byte is either Bowser (0x2D) or a flag (0x31)
            # this is to prevent returning true when Mario is using a vine
            # which will set the byte at 0x001D to 3
            if self.ram[address] in _STAGE_OVER_ENEMIES:
                # player float state set to 3 when sliding down flag pole
                return self.ram[0x001D] == 3

        return False

    @property
    def _flag_get(self):
        """Return a boolean determining if the agent reached a flag."""
        return self._is_world_over or self._is_stage_over

    # MARK: RAM Hacks

    def _write_stage(self):
        """Write the stage data to RAM to overwrite loading the next stage."""
        self.ram[0x075f] = self._target_world - 1
        self.ram[0x075c] = self._target_stage - 1
        self.ram[0x0760] = self._target_area - 1

    def _runout_prelevel_timer(self):
        """Force the pre-level timer to 0 to skip frames during a death."""
        self.ram[0x07A0] = 0

    def _skip_change_area(self):
        """Skip change area animations by by running down timers."""
        change_area_timer = self.ram[0x06DE]
        if change_area_timer > 1 and change_area_timer < 255:
            self.ram[0x06DE] = 1

    def _skip_occupied_states(self):
        """Skip occupied states by running out a timer and skipping frames."""
        while self._is_busy or self._is_world_over:
            self._runout_prelevel_timer()
            self._frame_advance(0)

    def _skip_start_screen(self):
        """Press and release start to skip the start screen."""
        # press and release the start button
        self._frame_advance(8)
        self._frame_advance(0)
        # Press start until the game starts
        while self._time == 0:
            # press and release the start button
            self._frame_advance(8)
            # if we're in the single stage, environment, write the stage data
            # if self.is_single_stage_env:
            #     self._write_stage()
            self._frame_advance(0)
            # run-out the prelevel timer to skip the animation
            self._runout_prelevel_timer()
        # set the last time to now
        self._time_last = self._time
        # after the start screen idle to skip some extra frames
        while self._time >= self._time_last:
            self._time_last = self._time
            self._frame_advance(8)
            self._frame_advance(0)

    def _skip_end_of_world(self):
        """Skip the cutscene that plays at the end of a world."""
        if self._is_world_over:
            # get the current game time to reference
            time = self._time
            # loop until the time is different
            while self._time == time:
                # frame advance with NOP
                self._frame_advance(0)

    def _kill_mario(self):
        """Skip a death animation by forcing Mario to death."""
        # force Mario's state to dead
        self.ram[0x000e] = 0x06
        # step forward one frame
        self._frame_advance(0)

    # MARK: Reward Function

    @property
    def _x_reward(self):
        """Return the reward based on left right movement between steps."""
        _reward = self._x_position - self._x_position_last
        self._x_position_last = self._x_position
        # TODO: check whether this is still necessary
        # resolve an issue where after death the x position resets. The x delta
        # is typically has at most magnitude of 3, 5 is a safe bound
        if _reward < -5 or _reward > 5:
            return 0

        return _reward

    @property
    def _time_penalty(self):
        """Return the reward for the in-game clock ticking."""
        _reward = self._time - self._time_last
        self._time_last = self._time
        # time can only decrease, a positive reward results from a reset and
        # should default to 0 reward
        if _reward > 0:
            return 0

        return _reward

    @property
    def _death_penalty(self):
        """Return the reward earned by dying."""
        if self._is_dying or self._is_dead:
            return -25

        return 0

    def step(self, action):
        """
        Step the environment with the given action.
        
        Args:
            action: The action can be either:
                   - An integer representing the action directly
                   - A string formatted like "[r][a]" to perform right+A
        
        Returns:
            A dictionary containing game state information
        """
        # Start timing this frame
        # super().step(0b00000000)
        frame_start_time = time.time()
        
        # If action is a string, parse it
        if isinstance(action, str):
            action = self.parse_action_string(action)
        
        # Perform the action in the environment
        observation, reward, done, info = super().step(action)
        
        # Update the total reward
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
            "text": (
                "Please provide your next action(s) in squared brackets. "
                "Any text you return that is not in squared brackets will "
                "be shown to you after, but not submitted to the environment. "
                "If you don't return any actions in squared brackets, no actions are executed."
            )
        }
        obs = {
            "visual": observation,
            "text": "Which buttons would you like to press?"
        }
        return obs, done, info

    # def step(self, action):
    #     """
    #     Step the environment with the given action.
        
    #     Args:
    #         action: The action can be either:
    #                - An integer representing the action directly
    #                - A string formatted like "[r][a]" to perform right+A
        
    #     Returns:
    #         A dictionary containing:
    #         - 'visual': The game screen as a serializable format
    #         - 'text': Formatted text information about the game state
    #         - Or a tuple of (observation, reward, done, info) if compatible_output=True
    #     """
    #     # Start timing this frame
    #     frame_start_time = time.time()
        
    #     # If action is a string, parse it
    #     if isinstance(action, str):
    #         action = self.parse_action_string(action)
        
    #     # Perform the action in the environment
    #     observation, reward, done, info = super().step(action)
        
    #     # Update the total reward
    #     self.total_reward += reward
        
    #     # Add additional information to the info dict
    #     info.update({
    #         'total_reward': self.total_reward,
    #         'speed_mode': self.speed_mode,
    #         'last_action': self.last_action_info,
    #     })
        
    #     # Throttle the frame rate based on the selected speed mode
    #     fps_info = self._throttle_fps()
    #     info.update(fps_info)
        
    #     obs = {
    #         "visual": observation,
    #         "text": "Go win!"
    #     }
    #     return obs, done, info
    #     # Convert numpy array to a serializable format
    #     # If observation is a numpy array, convert it to a list or another serializable format
    #     if isinstance(observation, np.ndarray):
    #         # Create a dictionary output format (similar to your original implementation)
    #         formatted_output = {
    #             'visual': observation.tolist(),  # Convert numpy array to list
    #             'text': self._get_formatted_info_text()
    #         }
    #         return formatted_output, done, info
        
    #     return observation,  done, info

    def _get_formatted_info_text(self):
        """Format the game information into a readable text string."""
        fps_info = self._throttle_fps()
        text = f"""
World: {self._world}-{self._stage}
Position: ({self._x_position}, {self._y_position})
Coins: {self._coins} | Lives: {self._life} | Time: {self._time}
Total Reward: {self.total_reward:.2f}
Mode: {self.speed_mode} ({self.FPS[self.speed_mode]} FPS)
Frame Time: {fps_info['frame_time_ms']:.1f}ms, Sleep: {fps_info['sleep_time_ms']:.1f}ms
{self.last_action_info}
"""
        return text

    def reset(self):
        """Reset the environment and return the initial observation."""
        # Reset the frame timer
        self.last_frame_time = time.time()
        
        # Reset tracking variables
        self._time_last = 0
        self._x_position_last = 0
        self.total_reward = 0
        self.last_action_info = "No actions executed yet."
        
        # Reset the environment
        observation = super().reset()
        
        return observation

    # MARK: nes-py API calls

    def _will_reset(self):
        """Handle and RAM hacking before a reset occurs."""
        self._time_last = 0
        self._x_position_last = 0

    def _did_reset(self):
        """Handle any RAM hacking after a reset occurs."""
        self._time_last = self._time
        self._x_position_last = self._x_position

    def _did_step(self, done):
        """
        Handle any RAM hacking after a step occurs.

        Args:
            done: whether the done flag is set to true

        Returns:
            None

        """
        # if done flag is set a reset is incoming anyway, ignore any hacking
        if done:
            return
        # if mario is dying, then cut to the chase and kill hi,
        if self._is_dying:
            self._kill_mario()
        # skip world change scenes (must call before other skip methods)
        # if not self.is_single_stage_env:
        #     self._skip_end_of_world()
        # skip area change (i.e. enter pipe, flag get, etc.)
        self._skip_change_area()
        # skip occupied states like the black screen between lives that shows
        # how many lives the player has left
        self._skip_occupied_states()

    def _get_reward(self):
        """Return the reward after a step occurs."""
        return self._x_reward + self._time_penalty + self._death_penalty

    def _get_done(self):
        """Return True if the episode is over, False otherwise."""
        # if self.is_single_stage_env:
        #     return self._is_dying or self._is_dead or self._flag_get
        return self._is_game_over

    def _get_info(self):
        """Return the info after a step occurs"""
        return dict(
            coins=self._coins,
            flag_get=self._flag_get,
            life=self._life,
            score=self._score,
            stage=self._stage,
            status=self._player_status,
            time=self._time,
            world=self._world,
            x_pos=self._x_position,
            y_pos=self._y_position,
            total_reward=self.total_reward,
            speed_mode=self.speed_mode,
            fps=self.FPS[self.speed_mode],
            last_action=self.last_action_info
        )
    
    def get_action_instructions(self):
        """Return formatted instructions for the user on how to use actions."""
        return ""
        return """
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


# explicitly define the outward facing API of this module
__all__ = [SuperMarioBrosEnv.__name__]