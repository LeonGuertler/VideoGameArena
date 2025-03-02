# import base64
# import requests
# import json
# import time
# import numpy as np
# from PIL import Image
# import io, os

# class OpenRouterAgent:
#     """
#     A wrapper to allow language models from OpenRouter API to play the game.
#     This handles sending observations and processing actions returned by the model.
#     """
    
#     def __init__(
#         self, 
#         model_name, 
#         system_prompt=None,
#         max_tokens=1024,
#         temperature=0.7
#     ):
#         """
#         Initialize the OpenRouter wrapper.
        
#         Args:
#             model (str): Model to use (default: anthropic/claude-3-opus)
#             system_prompt (str): System prompt to use (default: None)
#             max_tokens (int): Maximum tokens to generate (default: 1024)
#             temperature (float): Temperature for generation (default: 0.7)
#         """
#         self.model = model_name
#         self.max_tokens = max_tokens
#         self.temperature = temperature
        
#         # Set up default system prompt if none provided
#         if system_prompt is None:
#             self.system_prompt = """
#             You are playing Super Mario Bros. Your goal is to control Mario to complete levels.
#             You can see the game screen and receive information about the game state.
            
#             You must respond with actions to control Mario using the [x] format (e.g., [r] to move right).
#             You can combine actions like [r] [a] to jump right.
#             You can also sequence actions with + like [r] + [r] + [r] [a] to move right twice then jump.
            
#             Focus on making progress in the level. Respond ONLY with your action choices and brief reasoning.
#             """
#         else:
#             self.system_prompt = system_prompt
        
#         # Initialize conversation history
#         self.conversation_history = []


#         # load api key 
#         self.api_key = os.getenv("OPENROUTER_API_KEY")
#         if not self.api_key:
#             raise ValueError("OpenRouter API key not found. Please set the OPENROUTER_API_KEY environment variable.")
        
        
#     def _encode_image(self, image_array):
#         """
#         Convert a numpy image array to a base64 encoded string.
        
#         Args:
#             image_array (numpy.ndarray): The image as a numpy array
            
#         Returns:
#             str: Base64 encoded image string
#         """
#         # Convert numpy array to PIL Image
#         image = Image.fromarray(image_array.astype(np.uint8))
        
#         # Save image to bytes buffer
#         buffer = io.BytesIO()
#         image.save(buffer, format="PNG")
        
#         # Encode to base64
#         img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
#         return img_str
        
#     def __call__(self, observation, reward=None, done=None, info=None):
#         """
#         Process the observation, send it to the language model, and return the action.
        
#         Args:
#             observation (dict): Observation from the environment with 'visual' and 'text' keys
#             reward (float, optional): Reward from the last action
#             done (bool, optional): Whether the episode is done
#             info (dict, optional): Additional information from the environment
            
#         Returns:
#             str: Action string from the language model
#         """
#         # Prepare the model input
#         visual_data = observation.get('visual')
#         text_data = observation.get('text', '')
        
#         # Add game state information if available
#         if info:
#             text_data += f"\n\nGame State:\n"
#             text_data += f"World: {info.get('world', 'Unknown')}-{info.get('stage', 'Unknown')}\n"
#             text_data += f"Position: ({info.get('x_pos', 'Unknown')}, {info.get('y_pos', 'Unknown')})\n"
#             text_data += f"Coins: {info.get('coins', 0)} | Lives: {info.get('life', 0)} | Time: {info.get('time', 0)}\n"
            
#             # Add last action information if available
#             if 'last_action' in info:
#                 text_data += f"Last Action: {info['last_action']}\n"
            
#             # Add reward information if available
#             if reward is not None:
#                 text_data += f"Reward: {reward}\n"
            
#             # Add done information if available
#             if done is not None:
#                 text_data += f"Episode Done: {done}\n"
        
#         # Encode the image if available
#         content = []
#         if visual_data is not None:
#             # Base64 encode the image
#             encoded_image = self._encode_image(visual_data)
            
#             # Add the image to the content
#             content.append({
#                 "type": "image_url",
#                 "image_url": {
#                     "url": f"data:image/png;base64,{encoded_image}"
#                 }
#             })
        
#         # Add the text description
#         content.append({
#             "type": "text",
#             "text": text_data
#         })
        
#         # Add to conversation history
#         if len(self.conversation_history) > 5:
#             # Keep only the last 5 exchanges to avoid context length issues
#             self.conversation_history = self.conversation_history[-5:]
        
#         # Add the current observation to history
#         self.conversation_history.append({
#             "role": "user",
#             "content": content
#         })
        
#         # Prepare the API request
#         headers = {
#             "Content-Type": "application/json",
#             "Authorization": f"Bearer {self.api_key}"
#         }
        
#         payload = {
#             "model": self.model,
#             "messages": [
#                 {"role": "system", "content": self.system_prompt},
#                 *self.conversation_history
#             ],
#             "max_tokens": self.max_tokens,
#             "temperature": self.temperature
#         }
        
#         # Make the API request
#         try:
#             response = requests.post(
#                 "https://openrouter.ai/api/v1/chat/completions",
#                 headers=headers,
#                 json=payload
#             )
#             response.raise_for_status()
            
#             # Parse the response
#             response_data = response.json()
#             # print(response_data)
#             # print(response_data["choices"][0]["message"]["content"].strip())

#             model_response = response_data["choices"][0]["message"]["content"].strip()
            
#             #response_data["choices"][0].message.content.strip()
            
#             # Add the model's response to the conversation history
#             # self.conversation_history.append({
#             #     "role": "assistant",
#             #     "content": model_response
#             # })
            
#             return model_response
            
#         except requests.exceptions.RequestException as e:
#             print(f"Error making request to OpenRouter API: {e}")
#             # Return a safe default action
#             return "[n]"  # No operation
#         except (KeyError, IndexError) as e:
#             print(f"Error parsing API response: {e}")
#             return "[n]"  # No operation
    
#     def reset(self):
#         """Reset the conversation history."""
#         self.conversation_history = []


import os
import re
import gym
import time
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
import pygame

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

# class HumanAgent(Agent):
#     """Human agent class that directly captures keyboard input."""
    
#     def __init__(self):
#         """Initialize the human agent."""
#         super().__init__()
#         # Initialize pygame for key handling
#         pygame.init()
#         # Create a small window to capture key events
#         self.screen = pygame.display.set_mode((1, 1))
#         pygame.display.set_caption("Key Capture Window")
        
#         # Define key mappings
#         self.key_mapping = {
#             pygame.K_UP: 'u',     # Up
#             pygame.K_DOWN: 'd',   # Down
#             pygame.K_LEFT: 'l',   # Left
#             pygame.K_RIGHT: 'r',  # Right
#             pygame.K_a: 'a',      # Jump (A)
#             pygame.K_b: 'b',      # Run (B)
#         }
        
#     def __call__(self, observation: Union[str, Dict]) -> str:
#         """
#         Process keyboard input and return the action string.
        
#         Args:
#             observation: The observation from the environment.
            
#         Returns:
#             str: The action string based on active keys.
#         """
#         # Process pygame events
#         pygame.event.pump()
        
#         # Get currently pressed keys
#         pressed_keys = pygame.key.get_pressed()
        
#         # Convert pressed keys to action strings
#         # actions = []
#         actions = ""
#         for key_code, action_code in self.key_mapping.items():
#             if pressed_keys[key_code]:
#                 actions += f" [{action_code}] "
#                 # actions.append(f"[{action_code}]")
        
#         # If no keys are pressed, return no-op
#         if actions == "":
#             return "[n]"
        
#         print(actions)
#         return " ".join(actions)

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
            # pygame.K_w: 'u',      # Up alternative
            pygame.K_DOWN: 'd',   # Down
            # pygame.K_s: 'd',      # Down alternative
            pygame.K_LEFT: 'l',   # Left
            # pygame.K_a: 'l',      # Left alternative
            pygame.K_RIGHT: 'r',  # Right
            # pygame.K_d: 'r',      # Right alternative
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
# class HumanAgent(Agent):
#     """Human agent class that directly captures keyboard input using terminal input."""
    
#     def __init__(self):
#         """Initialize the human agent."""
#         super().__init__()
#         self.controls_info = """
# Controls:
# - Up arrow or W: [u]
# - Down arrow or S: [d]
# - Left arrow or A: [l]
# - Right arrow or D: [r]
# - Z: Jump [a]
# - X: Run [b]
# - Spacebar: No Operation [n]
# - Combinations work too (e.g., D+Z for jump right)

# Enter your action (e.g., [r] [a] for jump right): 
# """
    
#     def __call__(self, observation: Union[str, Dict]) -> str:
#         """
#         Get action input from the user through terminal.
        
#         Args:
#             observation: The observation from the environment.
            
#         Returns:
#             str: The action string based on user input.
#         """
#         # Display game info if available
#         if isinstance(observation, dict) and 'text' in observation:
#             print("\nGame Info:")
#             print(observation['text'])
        
#         # Get user input
#         user_input = input(self.controls_info)
        
#         # If no valid input, return no-op
#         if not re.search(r'\[[a-z]\]', user_input):
#             print("Using no-op action [n]")
#             return "[n]"
        
#         print(f"Action: {user_input}")
#         return user_input


class OpenRouterAgent(Agent):
    """Agent class using the OpenRouter API to generate responses."""
    
    STANDARD_GAME_PROMPT = """
    You are playing Super Mario Bros. Control Mario to complete the level.
    Use the [x] action format to control Mario.
    
    Available actions:
    - [a]: Jump (A button)
    - [b]: Run (B button)
    - [u]: Move up (for climbing)
    - [d]: Move down (for pipes)
    - [l]: Move left
    - [r]: Move right
    - [n]: No operation
    
    You can combine actions: [r] [a] (jump right)
    You can sequence actions: [r] + [r] + [r] [a] (move right twice, then jump right)
    
    Focus on making progress through the level.
    """
    
    def __init__(self, model_name: str, system_prompt: Optional[str] = None, verbose: bool = False, **kwargs):
        """
        Initialize the OpenRouter agent.

        Args:
            model_name (str): The name of the model.
            system_prompt (Optional[str]): The system prompt to use.
            verbose (bool): If True, additional debug info will be printed.
            **kwargs: Additional keyword arguments to pass to the OpenAI API call.
        """
        super().__init__()
        self.model_name = model_name 
        self.verbose = verbose 
        self.system_prompt = system_prompt or self.STANDARD_GAME_PROMPT
        self.kwargs = kwargs

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
                
                # Save image to bytes buffer
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                
                # Encode to base64
                img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                # Prepare the content for multimodal models
                content = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_str}"
                        }
                    },
                    {
                        "type": "text",
                        "text": observation.get('text', '')
                    }
                ]
            else:
                # Text-only observation
                content = observation.get('text', '')
        else:
            # Text-only observation
            content = observation
        
        # Prepare messages for the API
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
                    print(f"\nResponse: {response}")
                return response

            except Exception as e:
                last_exception = e
                print(f"Attempt {attempt} failed with error: {e}")
                if attempt < retries:
                    time.sleep(delay)
        raise last_exception

    def __call__(self, observation: Union[str, Dict]) -> str:
        """
        Process the observation using the OpenRouter API and return the action.

        Args:
            observation: The input to process.

        Returns:
            str: The generated response.
        """
        return self._retry_request(observation)