
import os, re, time
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
import pygame

import os
import re
import time
import threading
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
import queue
import matplotlib.pyplot as plt
from PIL import Image


class Agent(ABC):
    """Abstract base class for agents interacting with the environment."""
    
    def __init__(self):
        """Initialize the agent."""
        pass
    
    @abstractmethod
    def __call__(self, observation: Union[str, Dict]) -> str:
        """
        Process the observation and return the action.
        
        Args:
            observation: The observation from the environment.
            
        Returns:
            str: The action to take.
        """
        pass


class HumanAgent:
    """Human agent class that captures keyboard input via pygame."""
    
    def __init__(self):
        """Initialize the human agent."""
        # Initialize pygame for key handling
        pygame.init()
        # Create a window to capture key events
        self.screen = pygame.display.set_mode((320, 240))
        pygame.display.set_caption("Mario Control Window - Focus here to control the game")
        
        # Define basic key mappings
        self.key_mapping = {
            pygame.K_UP: 'u',     # Up
            pygame.K_DOWN: 'd',   # Down
            pygame.K_LEFT: 'l',   # Left
            pygame.K_RIGHT: 'r',  # Right
            pygame.K_a: 'a',      # Jump (A)
            pygame.K_b: 'b',      # Run (B)
            pygame.K_p: 'o',      # select
            pygame.K_o: 'p',      # start


        }
        
        # Draw control instructions on the window
        self.draw_controls()
    
    def draw_controls(self):
        """Draw control instructions on the pygame window."""
        self.screen.fill((0, 0, 0))  # Black background
        font = pygame.font.Font(None, 24)
        
        instructions = [
            "CONTROLS:",
            "Arrow keys or WASD: Movement",
            "Z: Jump (A button)",
            "X: Run (B button)",
            "",
            "Keep this window in focus",
            "to control Mario"
        ]
        
        y_pos = 30
        for line in instructions:
            text_surface = font.render(line, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(160, y_pos))
            self.screen.blit(text_surface, text_rect)
            y_pos += 30
        
        pygame.display.flip()
    
    def __call__(self, observation):
        """
        Process keyboard input and return the action string with all pressed keys.
        
        Args:
            observation: The observation from the environment.
            
        Returns:
            str: The action string based on all currently pressed keys.
        """
        # Process pygame events to keep the window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit("Game window closed")
            
            # Redraw if window is resized or needs update
            if event.type == pygame.VIDEORESIZE:
                self.draw_controls()
        
        # Get currently pressed keys
        pressed_keys = pygame.key.get_pressed()
        
        # Convert pressed keys to action strings
        actions = []
        for key_code, action_code in self.key_mapping.items():
            if pressed_keys[key_code]:
                actions.append(f"[{action_code}]")
        
        # If no keys are pressed, return no-op
        if not actions:
            # return "[start]"
            return "[n]"
        
        action_str = " ".join(actions)
        print(f"Action: {action_str}")
        return action_str

        



class OpenRouterAgent(Agent):
    """Agent class using the OpenRouter API to generate responses in a continuous background loop."""
    
    # STANDARD_GAME_PROMPT = """
    # You are playing Super Mario Bros. Control Mario to complete the level.
    # Use the [x] action format to control Mario.
    
    # Available actions:
    # - [a]: Jump (A button)
    # - [b]: Run (B button)
    # - [u]: Move up (for climbing)
    # - [d]: Move down (for pipes)
    # - [l]: Move left
    # - [r]: Move right
    # - [n]: No operation
    
    # You can combine actions: [r] [a] (jump right)
    
    # Focus on making progress through the level.
    # Respond ONLY with actions in the [x] format.
    # If you're not sure what to do, just return [n] for no operation.
    # Only submit the action for the buttons you want to press in this step. You cannot return lists of actions.
    # For example, you can press '[a] [r]' you can jump over enemies.
    # """
    STANDARD_GAME_PROMPT = """
    You are playing Super Mario Bros. Control Mario to complete the level.
    Use the [x] action format to control Mario.
    
    Available actions:
    - [a]: Jump (A button)
    - [l]: Move left
    - [r]: Move right
    - [n]: No operation
    
    You can combine actions: [r] [a] (jump right)
    
    Focus on making progress through the level.
    For example, you can press '[a] [r]' you can jump over enemies.
    """
    # STANDARD_GAME_PROMPT = """
    # You are playing Super Mario Bros. Control Mario to complete the level.
    # Use the [x] action format to control Mario.
    
    # Available actions:
    # - '[jump]'
    # - '[right]'
    # - '[left]'
    # - '[none]'
    
    # You can combine actions: '[rump] [right]' (jump right)
    
    # Focus on making progress through the level.
    # """
    # If you hold [a], mario will make one big jumo. To jump again, you need to let go of [a] first.
    # You will need to submit one action without [a] to be able to jump again. For example '[r]'.
    
    def __init__(self, model_name: str, system_prompt: Optional[str] = None, 
                 verbose: bool = False, update_interval: float = 0.1, **kwargs):
        """
        Initialize the OpenRouter agent with continuous background processing.

        Args:
            model_name (str): The name of the model.
            system_prompt (Optional[str]): The system prompt to use.
            verbose (bool): If True, additional debug info will be printed.
            update_interval (float): How often to check for new observations (seconds).
            **kwargs: Additional keyword arguments to pass to the OpenAI API call.
        """
        super().__init__()
        self.model_name = model_name 
        self.verbose = verbose 
        self.system_prompt = system_prompt or self.STANDARD_GAME_PROMPT
        self.kwargs = kwargs
        self.update_interval = update_interval
        
        # Initialize the current action to no-op
        self.current_action = "[n]"
        self.just_updated = False
        
        # Set up threading components
        self.observation_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.processing_thread = None
        self.is_thinking = False
        self.prev_img = None

        self.prev_actions = ""
        
        # Valid action keys for parsing responses
        self.valid_actions = ['a', 'b', 'u', 'd', 'l', 'r', 'n', 'o', 'p']
        self.action_history = "Action History: "
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package is required for OpenRouterAgent. "
                "Install it with: pip install openai"
            )

        # Set the open router api key from an environment variable
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API key not found. Please set the OPENROUTER_API_KEY environment variable.")
        
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        
        # Start the background processing thread
        self._start_background_thread()
        
    def _start_background_thread(self):
        """Start the background thread for continuous processing."""
        self.processing_thread = threading.Thread(target=self._background_processing_loop, daemon=True)
        self.processing_thread.start()
        if self.verbose:
            print("Background processing thread started")
        
    def _background_processing_loop(self):
        """Continuously process observations in the background."""
        while not self.stop_event.is_set():
            try:
                # Check if there's a new observation in the queue (non-blocking)
                try:
                    observation = self.observation_queue.get(block=False)
                    if self.verbose:
                        print("New observation received")
                    
                    # Set thinking flag
                    self.is_thinking = True
                    
                    # Process the observation and update current_action
                    try:
                        raw_response = self._retry_request(observation)
                        action_str = self._parse_actions(raw_response)
                        print(action_str)
                        self.action_history += f"\n{action_str}"
                        # Update the current action
                        if action_str:
                            # each "button" can at most be pressed once
                            actual_action_str = ""
                            # if "[jump]" in action_str or "[up]" in action_str:
                            #     actual_action_str += "[a]"
                            # if "[right]" in action_str:
                            #     actual_action_str += "[r]"
                            # if "[left]" in action_str:
                            #     actual_action_str += "[l]"
                            # if "[none]" in action_str:
                            #     actual_action_str += "[n]"
                            for la in self.valid_actions:
                                if la in action_str and la not in actual_action_str:
                                    actual_action_str += f"[{la}]"
                            action_str = actual_action_str
                            print(f"Action str: {action_str}")
                            self.current_action = action_str
                            self.just_updated = True
                        else:
                            self.current_action = "[n]"
                            
                        if self.verbose:
                            print(f"Updated action: {self.current_action}")
                    except Exception as e:
                        print(f"Error processing observation: {e}")
                        # Don't update current_action on error, keep using the previous one
                    
                    # Clear thinking flag
                    self.is_thinking = False
                    
                except queue.Empty:
                    # No new observation, continue loop
                    pass
                
                # Sleep briefly to avoid busy waiting
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"Error in background thread: {e}")
                # Sleep briefly to avoid spamming errors
                time.sleep(1)
    
    def stop(self):
        """Stop the background processing thread."""
        self.stop_event.set()
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
            if self.verbose:
                print("Background processing thread stopped")
    
    def _make_request(self, observation: Union[str, Dict]) -> str:
        """Make a single API request to OpenRouter and return the generated message."""
        # Process observation if it's a dictionary
        if isinstance(observation, dict):
            # If we have visual data, convert it to base64 for the API
            if 'visual' in observation:
                import base64
                from PIL import Image
                import io
                
                # Convert numpy array to PIL Image
                image = Image.fromarray(observation['visual'].astype(np.uint8))
                
                # print(image)
                # # Display the image inline
                # plt.imshow(image)
                # plt.axis('off')  # Hide axis if you like
                # plt.show()

                # Save image to bytes buffer
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                
                # Encode to base64
                img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
                # Prepare the content for multimodal models

                if self.prev_img is None:
                    self.prev_img = img_str
                content = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_str}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{self.prev_img}"
                        }
                    },
                    {
                        "type": "text",
                        "text": observation.get('text', '')
                        # "text": self.prev_actions + "\n" +observation.get('text', '')
                    }
                ]
                self.prev_img = img_str
            else:
                # Text-only observation
                content = observation.get('text', '')
        else:
            # Text-only observation
            content = observation
        
        # Prepare messages for the API
        print("#"*20)
        print(self.system_prompt)
        # print(content[0]["text"])
        print("#"*20)
        if isinstance(content, list):
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content}
            ]
        else:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content}
            ]
        
        # Make the API request
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            n=1,
            stop=None,
            **self.kwargs
        )

        self.prev_actions += f"\n{response.choices[0].message.content.strip()}"
        
        return response.choices[0].message.content.strip()

    def _retry_request(self, observation: Union[str, Dict], retries: int = 3, delay: int = 5) -> str:
        """
        Attempt to make an API request with retries.

        Args:
            observation: The input to process.
            retries (int): The number of attempts to try.
            delay (int): Seconds to wait between attempts.

        Raises:
            Exception: The last exception caught if all retries fail.
        """
        last_exception = None
        for attempt in range(1, retries + 1):
            try:
                response = self._make_request(observation)
                if self.verbose:
                    print(f"\nModel Output: {response}")
                return response

            except Exception as e:
                last_exception = e
                print(f"Attempt {attempt} failed with error: {e}")
                if attempt < retries:
                    time.sleep(delay)
        raise last_exception

    def _parse_actions(self, response_text: str) -> str:
        """
        Parse the model's response to extract valid action commands.
        
        Args:
            response_text (str): The raw response from the model
            
        Returns:
            str: Formatted action string with valid actions in [x] format
        """
        # Extract all actions in [x] format using regex
        action_groups = re.findall(r'\[(.[^\]]*)\]', response_text.lower())
        
        # Filter to valid actions only
        valid_action_groups = []
        for action_group in action_groups:
            # For each character in the action group, check if it's valid
            valid_chars = [c for c in action_group if c in self.valid_actions]
            if valid_chars:
                valid_action_groups.append(f"[{''.join(valid_chars)}]")
        
        # Join all valid actions with spaces
        return " ".join(valid_action_groups)

    def __call__(self, observation: Union[str, Dict]) -> str:
        """
        Process the observation using the OpenRouter API and return the action.
        This method only updates the observation queue and returns the current action.

        Args:
            observation: The input to process.

        Returns:
            str: The current action string.
        """
        # Update the observation queue (non-blocking, will replace any existing observation)
        try:
            # If queue is full, remove the old observation and add the new one
            if self.observation_queue.full():
                try:
                    self.observation_queue.get_nowait()
                except queue.Empty:
                    pass
            # Put the new observation in the queue
            self.observation_queue.put_nowait(observation)
        except queue.Full:
            if self.verbose:
                print("Queue is full, skipping observation update")
        
        # Add visual indicator if model is thinking
        if self.is_thinking and self.verbose:
            print("🤔 Thinking...")
        
        # if self.just_updated:
        #     self.just_updated = False
        #     return '[n]'
        # Always return the current action immediately
        return self.current_action