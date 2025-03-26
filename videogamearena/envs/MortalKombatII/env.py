import os
import re
import time
import numpy as np
import retro  # stable-retro


class MortalKombatIIEnv:
    """A simplified environment for Mortal Kombat II (Genesis) with stable-retro, supporting 2 players."""
    
    BUTTONS = ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
    BUTTON_MAP = {
        # Player 1 (lowercase)
        'u': 4, 'd': 5, 'l': 6, 'r': 7, 'a': 1, 'b': 0, 'c': 8, 'x': 10, 'y': 9, 'z': 11, 's': 3, 'm': 2, 'n': None,
        # Player 2 (uppercase, offset by +12)
        'U': 16, 'D': 17, 'L': 18, 'R': 19, 'A': 13, 'B': 12, 'C': 20, 'X': 22, 'Y': 21, 'Z': 23, 'S': 15, 'M': 14, 'N': None,
    }
    FPS = {'human': 60, 'slow': 30, 'super-slow': 10}

    def __init__(self, speed_mode='human', players=1, max_rounds=3):
        if speed_mode not in self.FPS:
            raise ValueError(f"Speed mode must be one of {list(self.FPS.keys())}")
        if players not in [1, 2]:
            raise ValueError("Players must be 1 or 2")
        
        self.players = players
        self.env = retro.make(game='MortalKombatII-Genesis', players=players)
        print(f"Successfully loaded MortalKombatII-Genesis with {players} player(s)")
        
        self.speed_mode = speed_mode
        self.target_frame_time = 1.0 / self.FPS[speed_mode]
        self.last_frame_time = time.time()

        self.last_action_info = "No actions executed yet."
        self.total_reward = 0
        self.max_rounds = max_rounds  # Best of N rounds (default 3)
        self.rounds_to_win = (max_rounds + 1) // 2  # e.g., 2 out of 3
        self.final_rewards = {0: 0, 1: 0}  # Dictionary to store final rewards for P1 (0) and P2 (1)
        
        self.reset()
        self._skip_start_screen()

    def set_speed_mode(self, mode):
        if mode not in self.FPS:
            raise ValueError(f"Speed mode must be one of {list(self.FPS.keys())}")
        self.speed_mode = mode
        self.target_frame_time = 1.0 / self.FPS[mode]
        print(f"Game speed set to {mode} mode ({self.FPS[mode]} FPS)")

    def _throttle_fps(self):
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
        print(f"Input action string: {action_string}")
        action_groups = re.findall(r'\[([^\]]*)\]', action_string)
        if not action_groups:
            self.last_action_info = "No actions executed (use [x] for P1, [X] for P2)."
            print("No valid actions found in string.")
            return [False] * (len(self.BUTTONS) * self.players)
        
        buttons = [False] * (len(self.BUTTONS) * self.players)
        for group in action_groups:
            for char in group:
                if char in self.BUTTON_MAP and self.BUTTON_MAP[char] is not None:
                    index = self.BUTTON_MAP[char]
                    buttons[index] = True
                    player = "P1" if char.islower() else "P2" if char.isupper() else "Unknown"
                    print(f"Set {player} button '{char}' at index {index}")
        
        self.last_action_info = f"Executed action: {action_string}"
        print(f"Resulting action array: {buttons}")
        return buttons

    def _skip_start_screen(self):
        start_action = [False] * (len(self.BUTTONS) * self.players)
        start_action[self.BUTTON_MAP['s']] = True  # P1 START
        if self.players == 2:
            start_action[self.BUTTON_MAP['S']] = True  # P2 START
        
        print(f"Skipping start screen with action: {start_action}")
        for _ in range(10):
            self.env.step(start_action)
            self.env.step([False] * (len(self.BUTTONS) * self.players))
            time.sleep(0.1)

    def step(self, action):
        if isinstance(action, str):
            action = self.parse_action_string(action)
        
        print(f"Stepping with action: {action}")
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Update round counts from info
        p1_rounds_won = info.get('rounds_won', 0)
        p2_rounds_won = info.get('enemy_rounds_won', 0)
        
        # Check if a round has ended
        round_ended = info.get('health', 100) <= 0 or info.get('enemy_health', 100) <= 0 or terminated
        
        # Override done: only True if someone wins best of max_rounds
        done = False
        if p1_rounds_won >= self.rounds_to_win:
            self.final_rewards = {0: 1, 1: -1}  # P1 wins
            reward = 1  # For P1 perspective during stepping
            done = True
            print(f"Match ended: P1 wins with {p1_rounds_won} rounds!")
        elif p2_rounds_won >= self.rounds_to_win:
            self.final_rewards = {0: -1, 1: 1}  # P2 wins
            reward = -1  # For P1 perspective during stepping
            done = True
            print(f"Match ended: P2 wins with {p2_rounds_won} rounds!")
        elif round_ended:
            print(f"Round ended. P1 rounds: {p1_rounds_won}, P2 rounds: {p2_rounds_won}")
            reward = 0  # No reward until match winner is determined
        
        self.total_reward += reward
        
        info.update({
            'total_reward': self.total_reward,
            'speed_mode': self.speed_mode,
            'last_action': self.last_action_info,
            'p1_health': info.get('health', 0),
            'p1_rounds_won': p1_rounds_won,
            'p2_health': info.get('enemy_health', 0),
            'p2_rounds_won': p2_rounds_won,
            'max_rounds': self.max_rounds,
            'rounds_to_win': self.rounds_to_win,
            'final_rewards': self.final_rewards
        })
        
        fps_info = self._throttle_fps()
        info.update(fps_info)
        
        obs = {
            "visual": observation,
            "text": "Which buttons would you like to press? (e.g., [u] for P1, [U] for P2)"
        }
        return obs, done, info

    def reset(self):
        self.last_frame_time = time.time()
        self.total_reward = 0
        self.last_action_info = "No actions executed yet."
        self.final_rewards = {0: 0, 1: 0}  # Reset final rewards
        observation = self.env.reset()
        return observation

    def render(self, mode='human'):
        return self.env.render()

    def close(self):
        print(f"Final Rewards: {self.final_rewards}")
        self.env.close()
        return self.final_rewards  # Return dictionary {0: reward_P1, 1: reward_P2}

    def get_action_instructions(self):
        return """
Action format: Submit actions in square brackets.
- Use lowercase [u] for Player 1.
- Use uppercase [U] for Player 2.

Available actions:
- [a] or [A]: Low Punch (A)
- [b] or [B]: Low Kick (B)
- [c] or [C]: Block (C)
- [x] or [X]: High Punch (X)
- [y] or [Y]: High Kick (Y)
- [u] or [U]: Up
- [d] or [D]: Down
- [l] or [L]: Left
- [r] or [R]: Right
- [s] or [S]: Start
- [m] or [M]: Select (MODE)
- [n] or [N]: No operation

Examples:
- [r][a]: P1 move right and low punch
- [L][X]: P2 move left and high punch
- [ra][DC]: P1 jump kick, P2 crouch block
"""