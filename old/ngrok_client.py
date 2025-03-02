import asyncio
import websockets
import cv2
import numpy as np
from pynput import keyboard
import json
import time
import logging
from collections import deque

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

URL = "ws://67f80d570dcf.ngrok.app"
# URL = "ws://localhost:8765"

class GameClient:
    def __init__(self, uri=URL, player_id=None):
        self.uri = uri
        self.websocket = None
        self.running = False
        self.key_state = {}  # Track pressed keys
        self.frame_times = deque(maxlen=100)  # For FPS calculation
        self.ping_times = deque(maxlen=10)    # For latency calculation
        self.last_frame_time = 0
        
        # Player information
        self.player_id = player_id  # Will be assigned by server if None
        self.game_mode = "unknown"
        self.current_player = None
        self.scores = {"player1": 0, "player2": 0}
        self.is_my_turn = False
        
        # Set up keyboard listener
        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release
        )
        
    def on_key_press(self, key):
        """Handle key press events."""
        key_str = str(key)
        self.key_state[key_str] = True
        return True
        
    def on_key_release(self, key):
        """Handle key release events."""
        key_str = str(key)
        if key_str in self.key_state:
            self.key_state[key_str] = False
        return True
    
    async def connect(self):
        """Connect to the game server."""
        try:
            self.websocket = await websockets.connect(self.uri)
            
            # Send initial connection message with player ID if we have one
            connect_message = {
                "type": "HUMAN",
                "player_id": self.player_id
            }
            await self.websocket.send(json.dumps(connect_message))
            logger.info(f"Connected to server at {self.uri}")
            
            # Wait for player assignment
            response = await self.websocket.recv()
            try:
                data = json.loads(response)
                if data.get("type") == "player_assignment":
                    self.player_id = data.get("player_id")
                    self.game_mode = data.get("game_mode", "unknown")
                    self.scores = data.get("scores", {"player1": 0, "player2": 0})
                    self.current_player = data.get("current_player")
                    self.is_my_turn = self.player_id == self.current_player
                    
                    logger.info(f"Assigned player ID: {self.player_id}")
                    logger.info(f"Game mode: {self.game_mode}")
                    logger.info(f"Current player: {self.current_player}")
                    
                    # Set window title to include player ID
                    cv2.namedWindow(f"Breakout - {self.player_id}", cv2.WINDOW_NORMAL)
                    return True
            except json.JSONDecodeError:
                # If not JSON, it's probably a frame - just store it for later processing
                self.first_frame = response
            
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
            
        # Filter to only keys that are pressed
        active_keys = {k: v for k, v in self.key_state.items() if v}
        try:
            await self.websocket.send(json.dumps({"keys": active_keys}))
        except Exception as e:
            logger.error(f"Failed to send key state: {e}")
    
    async def ping(self):
        """Measure server latency."""
        if not self.websocket:
            return
            
        try:
            start_time = time.time()
            await self.websocket.send(json.dumps({"ping": start_time}))
            response = await asyncio.wait_for(self.websocket.recv(), timeout=1.0)
            
            try:
                data = json.loads(response)
                if "pong" in data:
                    latency = (time.time() - float(data["pong"])) * 1000  # ms
                    self.ping_times.append(latency)
                    avg_latency = sum(self.ping_times) / len(self.ping_times)
                    logger.debug(f"Ping: {latency:.1f}ms (avg: {avg_latency:.1f}ms)")
                    return
            except (json.JSONDecodeError, ValueError):
                pass
                
            # If we get here, it's probably a frame - decode and display it
            await self.process_frame(response)
            
        except Exception as e:
            logger.warning(f"Ping failed: {e}")
    
    async def process_frame(self, frame_data):
        """Process and display a received frame."""
        try:
            # Calculate FPS
            current_time = time.time()
            if self.last_frame_time > 0:
                self.frame_times.append(current_time - self.last_frame_time)
            self.last_frame_time = current_time
            
            # Decode the frame (JPG format)
            nparr = np.frombuffer(frame_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                logger.error("Failed to decode image")
                return
                
            # Calculate and display FPS
            avg_frame_time = sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            
            # Calculate and display ping
            avg_ping = sum(self.ping_times) / len(self.ping_times) if self.ping_times else 0
            
            # Add FPS and ping information to the frame
            cv2.putText(
                img, 
                f"FPS: {fps:.1f} Ping: {avg_ping:.1f}ms", 
                (10, img.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                1
            )
            
            # Add player-specific information
            if self.game_mode == "competitive":
                # Add turn indicator
                turn_color = (0, 255, 0) if self.is_my_turn else (0, 0, 255)
                status_text = "YOUR TURN" if self.is_my_turn else "WAITING"
                
                cv2.putText(
                    img,
                    f"{status_text} ({self.player_id})",
                    (img.shape[1] // 2 - 80, img.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    turn_color,
                    2
                )
            
            # Add a "SLOW MOTION" indicator
            cv2.putText(
                img,
                "SLOW MOTION MODE",
                (img.shape[1] // 2 - 80, img.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                1
            )
            
            # Resize for better visibility
            img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
            
            # Display the frame
            cv2.imshow("Breakout", img)
            cv2.waitKey(1)  # Update display without blocking
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
    
    async def receive_frames(self):
        """Receive and process frames and game updates from the server."""
        if not self.websocket:
            return
            
        try:
            # Process first frame if we got one during connection
            if hasattr(self, 'first_frame'):
                await self.process_frame(self.first_frame)
                delattr(self, 'first_frame')
            
            while self.running:
                data = await self.websocket.recv()
                
                # Check if it's a JSON message
                try:
                    message = json.loads(data)
                    
                    # Handle different message types
                    if message.get("type") == "score_update":
                        self.scores = message.get("scores", self.scores)
                        player_scored = message.get("player_scored")
                        points = message.get("points", 0)
                        logger.info(f"Score update: {player_scored} scored {points} points")
                        
                    elif message.get("type") == "turn_change":
                        self.current_player = message.get("current_player")
                        self.is_my_turn = self.player_id == self.current_player
                        logger.info(f"Turn changed to: {self.current_player}")
                        
                        # Visual and audio feedback for turn change
                        if self.is_my_turn:
                            logger.info("It's your turn now!")
                            # Could add a sound effect here
                    
                    elif message.get("type") == "pong":
                        # Handle pong response (already handled in ping method)
                        pass
                        
                    continue  # Skip frame processing for JSON messages
                        
                except json.JSONDecodeError:
                    # Not JSON, must be a frame
                    pass
                
                # Process as a frame
                await self.process_frame(data)
                
        except Exception as e:
            logger.error(f"Frame reception error: {e}")
            self.running = False
    
    async def input_loop(self):
        """Send input to the server periodically."""
        try:
            while self.running:
                await self.send_key_state()
                await asyncio.sleep(1/30)  # Reduced to 30Hz to match server FPS
        except Exception as e:
            logger.error(f"Input loop error: {e}")
            self.running = False
    
    async def ping_loop(self):
        """Periodically measure server latency."""
        try:
            while self.running:
                await self.ping()
                await asyncio.sleep(1.0)  # Check ping every second
        except Exception as e:
            logger.error(f"Ping loop error: {e}")
    
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
                asyncio.create_task(self.receive_frames()),
                asyncio.create_task(self.input_loop()),
                asyncio.create_task(self.ping_loop())
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

# Run the client
async def main():
    client = GameClient()
    await client.run()

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Breakout Game Client")
    parser.add_argument("--player", type=str, help="Player ID (player1 or player2)")
    parser.add_argument("--server", type=str, default="ws://localhost:8765", help="Server WebSocket URL")
    args = parser.parse_args()
    
    # Create client with specified player ID
    client = GameClient(uri=args.server, player_id=args.player)
    
    # Run the client
    asyncio.run(client.run())