
import asyncio
import websockets
import cv2
import numpy as np
from pynput import keyboard
import json
import time
import logging
from collections import deque
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


URL = "ws://localhost:8765"
URL = "wss://67f80d570dcf.ngrok.app"


class GameClient:
    def __init__(self, uri=URL, player_id=None, game=None):
        self.uri = uri
        self.websocket = None
        self.running = False
        self.key_state = {}  # Track pressed keys
        self.frame_times = deque(maxlen=100)  # For FPS calculation
        self.ping_times = deque(maxlen=10)    # For latency calculation
        self.last_frame_time = 0
        
        # Game information
        self.player_id = player_id  # Will be assigned by server if None
        self.requested_game = game  # Game to request
        self.current_game = None    # Current game being played
        self.available_games = []   # List of available games
        self.is_agent = False       # Whether this client controls an agent
        self.agent = None           # Agent name if controlling one
        self.all_agents = []        # List of all agents
        self.scores = {}            # Current scores
        
        # Set up keyboard listener
        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release
        )
        
    def on_key_press(self, key):
        """Handle key press events."""
        key_str = str(key)
        self.key_state[key_str] = True
        
        # Handle special keys
        if key == keyboard.Key.f1:
            # F1 to cycle through available games
            asyncio.create_task(self.cycle_game())
            
        return True
        
    def on_key_release(self, key):
        """Handle key release events."""
        key_str = str(key)
        if key_str in self.key_state:
            self.key_state[key_str] = False
        return True
    
    async def cycle_game(self):
        """Request to change to the next available game."""
        if not self.available_games or not self.websocket:
            return
            
        # Find index of current game
        try:
            current_idx = self.available_games.index(self.current_game)
            next_idx = (current_idx + 1) % len(self.available_games)
        except ValueError:
            next_idx = 0
            
        next_game = self.available_games[next_idx]
        logger.info(f"Requesting game change to: {next_game}")
        
        try:
            await self.websocket.send(json.dumps({
                "change_game": next_game
            }))
        except Exception as e:
            logger.error(f"Failed to send game change request: {e}")
    
    async def connect(self):
        """Connect to the game server."""
        try:
            self.websocket = await websockets.connect(self.uri)
            
            # Send initial connection message
            connect_message = {
                "type": "HUMAN",
                "player_id": self.player_id,
                "game": self.requested_game
            }
            await self.websocket.send(json.dumps(connect_message))
            logger.info(f"Connected to server at {self.uri}")
            
            # Wait for game info
            response = await self.websocket.recv()
            try:
                data = json.loads(response)
                if data.get("type") == "game_info":
                    self.current_game = data.get("game")
                    self.player_id = data.get("player_id")
                    self.is_agent = data.get("is_agent", False)
                    self.agent = data.get("agent")
                    self.all_agents = data.get("all_agents", [])
                    self.available_games = data.get("available_games", [])
                    
                    logger.info(f"Game info received:")
                    logger.info(f"  Game: {self.current_game}")
                    logger.info(f"  Player ID: {self.player_id}")
                    logger.info(f"  Agent: {self.agent if self.is_agent else 'None'}")
                    logger.info(f"  Available games: {self.available_games}")
                    
                    # Set window title
                    window_title = f"{self.current_game} - {self.player_id}"
                    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
                    return True
            except json.JSONDecodeError:
                # If not JSON, it's probably a frame - just store it for later processing
                logger.info("Received binary data during connection")
                await self.handle_binary_frame(response)
            
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from the server."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            logger.info("Disconnected from server")
    
    async def send_key_state(self):
        """Send current key state to server."""
        if not self.websocket:
            return
            
        # Add ping information to the key state
        message = {
            "keys": self.key_state,
            "ping": time.time()  # Include ping timestamp with every key state update
        }
            
        try:
            await self.websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send key state: {e}")
    
    async def input_loop(self):
        """Send input to the server periodically."""
        try:
            while self.running:
                await self.send_key_state()
                await asyncio.sleep(1/30)  # 30Hz input sampling rate
        except Exception as e:
            logger.error(f"Input loop error: {e}")
            self.running = False
    
    async def receive_loop(self):
        """Receive messages from the server."""
        while self.running and self.websocket:
            try:
                # Wait for a message from the server
                message = await self.websocket.recv()
                
                # Check if the message is a string or bytes
                if isinstance(message, str):
                    try:
                        # Try to parse as JSON
                        data = json.loads(message)
                        self.handle_json_message(data)
                    except json.JSONDecodeError:
                        # Not valid JSON, but still a string - log and ignore
                        logger.warning(f"Received non-JSON string: {message[:50]}...")
                else:
                    # Binary data
                    await self.handle_binary_frame(message)
                    
            except websockets.ConnectionClosed:
                logger.error("Connection closed by server")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error in receive loop: {e}")
                await asyncio.sleep(0.1)  # Short delay to avoid tight error loops
    
    def handle_json_message(self, data):
        """Handle a JSON message from the server."""
        # Check for message type
        if "type" in data:
            msg_type = data["type"]
            
            if msg_type == "game_info":
                self.current_game = data.get("game")
                self.player_id = data.get("player_id")
                self.is_agent = data.get("is_agent", False)
                self.agent = data.get("agent")
                self.all_agents = data.get("all_agents", [])
                self.available_games = data.get("available_games", [])
                logger.info(f"Game info updated: {self.current_game}")
                
            elif msg_type == "game_changing":
                new_game = data.get("new_game")
                logger.info(f"Game changing to: {new_game}")
                # Close current window
                cv2.destroyAllWindows()
                # Create new window for the new game
                window_title = f"{new_game} - {self.player_id}"
                cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
                
            elif msg_type == "score_update":
                self.scores = data.get("scores", self.scores)
                player_scored = data.get("player_scored")
                reward = data.get("reward", 0)
                logger.info(f"Score update: {player_scored} scored {reward} points")
        
        # Check for ping response
        elif "pong" in data:
            ping_time = data.get("pong")
            if ping_time:
                latency = (time.time() - float(ping_time)) * 1000  # ms
                self.ping_times.append(latency)
    
    async def handle_binary_frame(self, frame_data):
        """Handle a binary frame (game screen) from the server."""
        try:
            # Calculate FPS
            current_time = time.time()
            if self.last_frame_time > 0:
                self.frame_times.append(current_time - self.last_frame_time)
            self.last_frame_time = current_time
            
            # Convert bytes to numpy array
            nparr = np.frombuffer(frame_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                logger.error("Failed to decode image")
                return
                
            # Calculate FPS and latency
            avg_frame_time = sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            avg_ping = sum(self.ping_times) / len(self.ping_times) if self.ping_times else 0
            
            # Add status information to the frame
            status_height = 30
            status_img = np.zeros((status_height, img.shape[1], 3), dtype=np.uint8)
            
            # Add FPS and ping information
            cv2.putText(
                status_img, 
                f"FPS: {fps:.1f} Ping: {avg_ping:.1f}ms", 
                (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                1
            )
            
            # Add player status
            status_text = f"{self.player_id}"
            if self.is_agent:
                status_text += f" (PLAYING)"
                status_color = (0, 255, 0)
            else:
                status_text += f" (SPECTATING)"
                status_color = (255, 165, 0)
                
            cv2.putText(
                status_img,
                status_text,
                (img.shape[1] // 2 - 60, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                status_color,
                1
            )
            
            # Add game controls help
            cv2.putText(
                status_img,
                "F1: Change Game",
                (img.shape[1] - 150, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
            # Combine status bar with game image
            display_img = np.vstack((status_img, img))
            
            # Resize for better visibility if small
            if display_img.shape[1] < 400:
                scale = 2
                display_img = cv2.resize(
                    display_img, 
                    (display_img.shape[1] * scale, display_img.shape[0] * scale), 
                    interpolation=cv2.INTER_NEAREST
                )
            
            # Get the correct window title
            window_title = f"{self.current_game} - {self.player_id}"
            
            # Display the frame
            cv2.imshow(window_title, display_img)
            cv2.waitKey(1)  # Update display without blocking
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
    
    async def run(self):
        """Run the game client."""
        # Start keyboard listener
        self.keyboard_listener.start()
        
        # Connect to server
        if not await self.connect():
            return
        
        try:
            self.running = True
            
            # Create tasks for receiving frames and sending input
            tasks = [
                asyncio.create_task(self.receive_loop()),
                asyncio.create_task(self.input_loop())
            ]
            
            # Wait for any task to complete (or error)
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except KeyboardInterrupt:
            logger.info("Client stopped by user")
        except Exception as e:
            logger.error(f"Client error: {e}")
        finally:
            # Clean up
            self.running = False
            self.keyboard_listener.stop()
            await self.disconnect()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Multiplayer Atari Game Client")
    parser.add_argument("--player", type=str, help="Player ID")
    parser.add_argument("--server", type=str, default="ws://localhost:8765", help="Server WebSocket URL")
    parser.add_argument("--game", type=str, help="Game to play (pong_v3, space_invaders_v2, joust_v3)")
    args = parser.parse_args()
    
    # Create client with specified parameters
    client = GameClient(
        uri=args.server, 
        player_id=args.player,
        game=args.game
    )
    
    # Run the client
    asyncio.run(client.run())