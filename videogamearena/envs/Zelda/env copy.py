# heavily based on / copied from https://github.com/Kautenja/gym-zelda-1/tree/master
import os, re, time, collections
from nes_py import NESEnv
import numpy as np


# the directory that houses this module
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


# a mapping of numeric values to cardinal directions
DIRECTIONS = collections.defaultdict(lambda: None, {
    0x08: 'N',
    0x04: 'S',
    0x01: 'E',
    0x02: 'W',
})


# the set of game modes that indicate a scroll is in progress
SCROLL_GAME_MODES = {4, 6, 7}


# a mapping of numeric values to string types for pulse 1
PULSE_1_IM_TYPES = collections.defaultdict(lambda: None, {
    0x80: None, # this value is unknown
    0x40: "1 Heart Warning",
    0x20: "Set Bomb",
    0x10: "Small Heart Pickup",
    0x08: "Key Pickup",
    0x04: "Magic Cast",
    0x02: "Boomerang Stun",
    0x01: "Arrow Deflected",
})


# a mapping of numeric values to string types for pulse 2
PULSE_2_IM_TYPES = collections.defaultdict(lambda: None, {
    0x80: "Death Spiral",
    0x40: "Continue Screen",
    0x20: "Enemy Burst",
    0x10: "Whistle",
    0x08: "Bomb Pickup",
    0x04: "Secret Revealed",
    0x02: "Key Appears",
    0x01: "Rupee Pickup",
})


# a mapping of numeric values to sword types
SWORD_TYPES = collections.defaultdict(lambda: None, {
    0x00: "None",
    0x01: "Sword",
    0x02: "White Sword",
    0x03: "Magical Sword",
})


# the type of arrows in Link's inventory
ARROWS_TYPES = collections.defaultdict(lambda: None, {
    0x00: "None",
    0x01: "Arrow",
    0x02: "Silver Arrow",
})


# the type of candle in Link's inventory
CANDLE_TYPES = collections.defaultdict(lambda: None, {
    0x00: "None",
    0x01: "Blue Candle",
    0x02: "Red Candle",
})


# the type of potion in Link's inventory
POTION_TYPES = collections.defaultdict(lambda: None, {
    0x00: "None",
    0x01: "Life Potion",
    0x02: "2nd Potion",
})


# the type of ring in Link's inventory
RING_TYPES = collections.defaultdict(lambda: None, {
    0x00: "None",
    0x01: "Blue Ring",
    0x02: "Red Ring",
})


class ZeldaEnv(NESEnv):
    """An environment for playing The Legend of Zelda with OpenAI Gym."""

    # the legal range of rewards for each step
    reward_range = (-float('inf'), float('inf'))

    # Byte mapping for input actions
    BYTE_MAPPING = {
        "u": 0b00010000,  # Up
        "d": 0b00100000,  # Down
        "l": 0b01000000,  # Left
        "r": 0b10000000,  # Right
        "a": 0b00000001,  # A button (sword)
        "b": 0b00000010,  # B button (item)
        "o": 0b00001000,  # start
        "p": 0b00000100,  # select
        "n": 0b00000000,  # no operation
    }

    # Legal input bytes (combinations allowed)
    LEGAL_BYTES = [
        0b10000001,  # ('r', 'a'), - sword right
        0b10000010,  # ('r', 'b'), - item right
        0b10000011,  # ('r', 'a', 'b'), - sword and item right
        0b01000001,  # ('l', 'a'), - sword left
        0b01000010,  # ('l', 'b'), - item left
        0b01000011,  # ('l', 'a', 'b'), - sword and item left
        0b00010001,  # ('u', 'a'), - sword up
        0b00010010,  # ('u', 'b'), - item up
        0b00010011,  # ('u', 'a', 'b'), - sword and item up
        0b00100001,  # ('d', 'a'), - sword down
        0b00100010,  # ('d', 'b'), - item down
        0b00100011,  # ('d', 'a', 'b'), - sword and item down
        0b00010000,  # up
        0b00100000,  # down
        0b01000000,  # left
        0b10000000,  # right
        0b00000001,  # A (sword)
        0b00000010,  # B (item)
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
        Initialize a new Zelda environment.

        Args:
            speed_mode (str): Game speed mode - 'human', 'slow', or 'super-slow'

        Returns:
            None
        """
        # Check if the speed mode is valid
        if speed_mode not in self.FPS:
            raise ValueError(f"Speed mode must be one of {list(self.FPS.keys())}")
        
        # Initialize with the ROM path
        super().__init__("videogamearena/envs/Zelda/Zelda_1.nes")
        
        # Setup frame rate control variables
        self.speed_mode = speed_mode
        self.target_frame_time = 1.0 / self.FPS[speed_mode]
        self.last_frame_time = time.time()
        
        # Setup UI tracking variables
        self.last_action_info = "No actions executed yet."
        self.total_reward = 0
        
        # reset the emulator, skip the start screen, and create a backup state
        self.reset()
        self._skip_start_screen()
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

    @property
    def _is_screen_scrolling(self):
        """Return True if the screen is scrolling, False otherwise."""
        return self.ram[0x12] in SCROLL_GAME_MODES

    @property
    def _current_level(self):
        """Return the current level Link is in."""
        return self.ram[0x10]

    @property
    def _current_save_slot(self):
        """Return the current save slot being played on."""
        return self.ram[0x16]

    @property
    def _x_pixel(self):
        """Return the current x pixel of Link's location."""
        return self.ram[0x70]

    @property
    def _y_pixel(self):
        """Return the current y pixel of Link's location."""
        return self.ram[0x84]

    @property
    def _direction(self):
        """Return the current direction that Link is facing."""
        return DIRECTIONS[self.ram[0x98]]

    @property
    def _has_candled(self):
        """Return True if Link has used a candle in the current room"""
        return bool(self.ram[0x0513])

    @property
    def _pulse_1_IM_type(self):
        """Return the IM type of pulse 1."""
        # TODO: gives "Small Heart" when text is blitting?
        return PULSE_1_IM_TYPES[self.ram[0x0605]]

    @property
    def _pulse_2_IM_type(self):
        """Return the IM type of pulse 2."""
        # TODO: gives "Bomb" when initial sword is picked up?
        return PULSE_2_IM_TYPES[self.ram[0x0607]]

    @property
    def _killed_enemy_count(self):
        """Return thee number of enemies killed on the current screen."""
        return self.ram[0x0627]

    @property
    def _number_of_deaths(self):
        """Return the number of times Link has died (for slot 1)."""
        # 0630    Number of deaths            save slot 1
        # 0631    Number of deaths            save slot 2
        # 0632    Number of deaths            save slot 3
        return self.ram[0x0630]

    @property
    def _sword(self):
        """Return the sword Link has."""
        return SWORD_TYPES[self.ram[0x0657]]

    @property
    def _number_of_bombs(self):
        """Return the number of bombs in inventory."""
        return self.ram[0x0658]

    @property
    def _arrows_type(self):
        """Return the type of arrows Link has."""
        return ARROWS_TYPES[self.ram[0x0659]]

    @property
    def _is_bow_in_inventory(self):
        """Return True if the bow is in Link's inventory."""
        return bool(self.ram[0x065A])

    @property
    def _candle_type(self):
        """Return the status of the candle Link has."""
        return CANDLE_TYPES[self.ram[0x065B]]

    @property
    def _is_whistle_in_inventory(self):
        """Return True if the candle is in Link's inventory."""
        return bool(self.ram[0x065C])

    @property
    def _is_food_in_inventory(self):
        """Return True if food is in Link's inventory."""
        return bool(self.ram[0x065D])

    @property
    def _potion_type(self):
        """Return True if potion is in Link's inventory."""
        return POTION_TYPES[self.ram[0x065E]]

    @property
    def _is_magic_rod_in_inventory(self):
        """Return True if the magic rod is in Link's inventory."""
        return bool(self.ram[0x065F])

    @property
    def _is_raft_in_inventory(self):
        """Return True if the raft is in Link's inventory."""
        return bool(self.ram[0x0660])

    @property
    def _is_magic_book_in_inventory(self):
        """Return True if the magic book is in Link's inventory."""
        return bool(self.ram[0x0661])

    @property
    def _ring_type(self):
        """Return True if the ring is in Link's inventory."""
        return RING_TYPES[self.ram[0x0662]]

    @property
    def _is_step_ladder_in_inventory(self):
        """Return True if the ladder is in Link's inventory."""
        return bool(self.ram[0x0663])

    @property
    def _is_magical_key_in_inventory(self):
        """Return True if the magic key is in Link's inventory."""
        return bool(self.ram[0x0664])

    @property
    def _is_power_bracelet_in_inventory(self):
        """Return True if the power bracelet is in Link's inventory."""
        return bool(self.ram[0x0665])

    @property
    def _is_letter_in_inventory(self):
        """Return True if the letter is in Link's inventory."""
        return bool(self.ram[0x0666])

    @property
    def _compass(self):
        """Return the mapping of which compasses are collected."""
        return self.ram[0x0667]
        # 0667    Compass in Inventory        One bit per level
        # 0669    Compass in Inventory        (Level 9)

    @property
    def _map(self):
        """Return the mapping of which maps are collected."""
        return self.ram[0x0668]
        # 0668    Map in Inventory            One bit per level
        # 066A    Map in Inventory            (Level 9)

    @property
    def _is_clock_possessed(self):
        """Return True if the clock is possessed."""
        return bool(self.ram[0x066C])

    @property
    def _number_of_rupees(self):
        """Return the number of rupees Link has."""
        return self.ram[0x066D]

    @property
    def _number_of_keys(self):
        """Return the number of keys Link has."""
        return self.ram[0x066E]

    @property
    def _number_of_heart_containers(self):
        """Return the number of total heart containers."""
        return (self.ram[0x066F] >> 4) + 1

    @property
    def _full_hearts_remaining(self):
        """Return the number of remaining hearts."""
        return 0x0F & self.ram[0x066F]

    @property
    def _partial_heart_remaining(self):
        """Return the amount of the partial heart remaining (percentage)."""
        return self.ram[0x0670] / 255

    @property
    def _hearts_remaining(self):
        """Return the amount of floating point remaining hears."""
        return self._full_hearts_remaining + self._partial_heart_remaining

    @property
    def _triforce_pieces(self):
        """Return the triforce pieces collected."""
        return self.ram[0x0671]
        # 0671 Triforce pieces. One bit per piece

    @property
    def _is_boomerang_in_inventory(self):
        """Return True if the boomerang is in Link's inventory."""
        return bool(self.ram[0x0674])

    @property
    def _is_magic_boomerang_in_inventory(self):
        """Return True if the magic boomerang is in Link's inventory."""
        return bool(self.ram[0x0675])

    @property
    def _is_magic_shield_in_inventory(self):
        """Return True if the magic shield is in Link's inventory."""
        return bool(self.ram[0x0676])

    @property
    def _max_number_of_bombs(self):
        """Return the max number of bombs that Link can carry."""
        return self.ram[0x067C]

    # MARK: RAM Hacks

    def _skip_start_screen(self):
        """Press and release start to skip the start screen."""
        # press and release the start button 21 times
        # - kill 21 frames to get to registration
        # - kill 10 frames to get to player 1 registration
        for _ in range(31):
            self._frame_advance(8)
            self._frame_advance(0)
        # select the letter A and kill 6 frames
        for _ in range(6):
            self._frame_advance(1)
            self._frame_advance(0)
        # move the cursor to the register button
        for _ in range(3):
            self._frame_advance(4)
            self._frame_advance(0)
        # press select to register the profile and subsequently start the game
        # by killing some frames and pressing select again
        for _ in range(9):
            self._frame_advance(8)
            self._frame_advance(0)
        # skip the opening screen animation
        while self._direction is None or bool(self.ram[0x007C]):
            self._frame_advance(0)

    def _wait_for_hearts(self):
        """Skip the death animation when Link dies."""
        while self._hearts_remaining <= 0:
            self._frame_advance(8)
            self._frame_advance(0)

    def _wait_for_scroll(self):
        """Wait for the screen to stop scrolling."""
        while self._is_screen_scrolling:
            self._frame_advance(8)
            self._frame_advance(0)

    def _skip_boring_actions(self):
        """Skip actions that the agent will find boring."""
        # displaying text
        while self.ram[0x0605] == 0x10:
            # each character takes 6 frames to draw
            for _ in range(6):
                self._frame_advance(0)
        # entering / exiting cave
        while self.ram[0x0606] == 0x08:
            self._frame_advance(0)

    def _skip_inventory_scroll(self):
        """Skip the scrolling action when showing / hiding inventory."""
        while 65 < self.ram[0xFC]:
            self._frame_advance(0)

    # MARK: Override step and reset methods

    def step(self, action):
        """
        Step the environment with the given action.
        
        Args:
            action: The action can be either:
                   - An integer representing the action directly
                   - A string formatted like "[r][a]" to perform right+sword
        
        Returns:
            A dictionary containing:
            - 'visual': The game screen as a serializable format
            - 'text': Formatted text information about the game state
            - Or a tuple of (observation, reward, done, info) if compatible_output=True
        """
        # Start timing this frame
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
        
        # Convert numpy array to a serializable format
        # If observation is a numpy array, convert it to a list or another serializable format
        if isinstance(observation, np.ndarray):
            # Create a dictionary output format
            formatted_output = {
                'visual': observation.tolist(),  # Convert numpy array to list
                'text': self._get_formatted_info_text()
            }
            return formatted_output, done, info
        
        return observation, reward, done, info
    
    def reset(self):
        """Reset the environment and return the initial observation."""
        # Reset the frame timer
        self.last_frame_time = time.time()
        
        # Reset tracking variables
        self.total_reward = 0
        self.last_action_info = "No actions executed yet."
        
        # Reset the environment
        observation = super().reset()
        
        # Create a dictionary with visual observation and instructions
        if isinstance(observation, np.ndarray):
            formatted_observation = {
                'visual': observation.tolist(),  # Convert numpy array to list
                'text': self.get_action_instructions() + "\n\n" + self._get_formatted_info_text()
            }
            return formatted_observation
        
        return observation

    def _get_formatted_info_text(self):
        """Format the game information into a readable text string."""
        fps_info = self._throttle_fps()
        
        # Get essential player statistics
        hearts_text = f"{self._full_hearts_remaining} + {self._partial_heart_remaining:.2f}/{self._number_of_heart_containers}"
        
        # Get inventory status
        inventory = []
        if self._sword != "None":
            inventory.append(self._sword)
        if self._is_bow_in_inventory:
            if self._arrows_type != "None":
                inventory.append(f"{self._arrows_type}s")
            else:
                inventory.append("Bow")
        if self._number_of_bombs > 0:
            inventory.append(f"{self._number_of_bombs} Bombs")
        if self._candle_type != "None":
            inventory.append(self._candle_type)
        if self._is_boomerang_in_inventory:
            inventory.append("Boomerang")
        if self._is_magic_boomerang_in_inventory:
            inventory.append("Magic Boomerang")
        if self._ring_type != "None":
            inventory.append(self._ring_type)
        if self._is_magic_shield_in_inventory:
            inventory.append("Magic Shield")
        
        text = f"""
Level: {self._current_level} | Direction: {self._direction or 'None'}
Position: ({self._x_pixel}, {self._y_pixel})
Hearts: {hearts_text} | Rupees: {self._number_of_rupees} | Keys: {self._number_of_keys}
Enemies Killed: {self._killed_enemy_count} | Deaths: {self._number_of_deaths}
Inventory: {', '.join(inventory) if inventory else 'Empty'}
Total Reward: {self.total_reward:.2f}
Mode: {self.speed_mode} ({self.FPS[self.speed_mode]} FPS)
Frame Time: {fps_info['frame_time_ms']:.1f}ms, Sleep: {fps_info['sleep_time_ms']:.1f}ms
{self.last_action_info}
"""
        return text
    
    def get_action_instructions(self):
        """Return formatted instructions for the user on how to use actions."""
        return """
Action format: Submit actions in square brackets like [a] or [r].
You can submit multiple actions simultaneously: [r] [a] (equivalent to [ra])
You can also submit action sequences: [r] + [r] + [ra] (move right twice, then attack right)

Available actions:
- [a]: Attack with sword (A button)
- [b]: Use selected item (B button)
- [u]: Move up
- [d]: Move down
- [l]: Move left
- [r]: Move right
- [o]: Open start menu/pause (Start button)
- [p]: Open item select (Select button)
- [n]: No operation

Common combinations:
- [ra]: Attack right
- [rb]: Use item to the right
- [ua]: Attack up
- [ub]: Use item upward
- [la]: Attack left
- [lb]: Use item to the left
- [da]: Attack down
- [db]: Use item downward
"""

    # MARK: nes-py API calls

    def _will_reset(self):
        """Handle and RAM hacking before a reset occurs."""
        pass

    def _did_reset(self):
        """Handle any RAM hacking after a reset occurs."""
        pass

    def _did_step(self, done):
        """
        Handle any RAM hacking after a step occurs.

        Args:
            done: whether the done flag is set to true

        Returns:
            None

        """
        # Only perform these skips if we're not done
        if not done:
            self._wait_for_hearts()
            self._wait_for_scroll()
            self._skip_boring_actions()
            self._skip_inventory_scroll()

    def _get_reward(self):
        """Return the reward after a step occurs."""
        # Basic reward system - could be enhanced
        reward = 0
        
        # Reward for killing enemies
        reward += self._killed_enemy_count * 0.5
        
        # Reward for collecting rupees
        reward += self._number_of_rupees * 0.1
        
        # Reward for having hearts
        reward += self._hearts_remaining * 0.2
        
        # Reward for exploring (can be modified further)
        reward += 0.01
        
        return reward

    def _get_done(self):
        """Return True if the episode is over, False otherwise."""
        # Episode is over if Link has no hearts left
        if self._hearts_remaining <= 0:
            return True
        
        # Could add more end conditions here
        return False

    def _get_info(self):
        """Return the info after a step occurs"""
        return dict(
            current_level=self._current_level,
            x_pos=self._x_pixel,
            y_pos=self._y_pixel,
            direction=self._direction,
            has_candled=self._has_candled,
            pulse_1=self._pulse_1_IM_type,
            pulse_2=self._pulse_2_IM_type,
            killed_enemies=self._killed_enemy_count,
            number_of_deaths=self._number_of_deaths,
            sword=self._sword,
            number_of_bombs=self._number_of_bombs,
            arrows_type=self._arrows_type,
            has_bow=self._is_bow_in_inventory,
            candle_type=self._candle_type,
            has_whistle=self._is_whistle_in_inventory,
            has_food=self._is_food_in_inventory,
            potion_type=self._potion_type,
            has_magic_rod=self._is_magic_rod_in_inventory,
            has_raft=self._is_raft_in_inventory,
            has_magic_book=self._is_magic_book_in_inventory,
            ring_type=self._ring_type,
            has_step_ladder=self._is_step_ladder_in_inventory,
            has_magic_key=self._is_magical_key_in_inventory,
            has_power_bracelet=self._is_power_bracelet_in_inventory,
            has_letter=self._is_letter_in_inventory,
            is_clock_possessed=self._is_clock_possessed,
            rupees=self._number_of_rupees,
            keys=self._number_of_keys,
            heart_containers=self._number_of_heart_containers,
            hearts=self._hearts_remaining,
            has_boomerang=self._is_boomerang_in_inventory,
            has_magic_boomerang=self._is_magic_boomerang_in_inventory,
            has_magic_shield=self._is_magic_shield_in_inventory,
            max_number_of_bombs=self._max_number_of_bombs,
            total_reward=self.total_reward,
            speed_mode=self.speed_mode,
            fps=self.FPS[self.speed_mode],
            last_action=self.last_action_info
        )


# explicitly define the outward facing API of this module
__all__ = [ZeldaEnv.__name__]