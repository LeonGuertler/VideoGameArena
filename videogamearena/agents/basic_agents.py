
import os, re, time
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