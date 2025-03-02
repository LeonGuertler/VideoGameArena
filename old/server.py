import asyncio
import websockets
import cv2
import numpy as np
import json
import time
import logging
import uuid
from collections import deque

# Import PettingZoo
try:
    from pettingzoo.atari import pong_v3, space_invaders_v2, joust_v3
except ImportError:
    logging.error("PettingZoo not installed. Please install with: pip install pettingzoo[atari]")
    exit(1)

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom JSON encoder to handle numpy types
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)

# Function to encode observation as JPG
def encode_obs(obs, game_info=None):
    """Convert observation to JPEG binary data with added game information overlay."""
    if obs is None:
        # Create a blank image if observation is None
        img = np.zeros((210, 160, 3), dtype=np.uint8)
    else:
        # Convert to BGR for OpenCV
        img = np.copy(obs)
    
    # Add game information overlay if provided
    if game_info:
        # Create a larger canvas to add info
        height, width = img.shape[:2]
        info_height = 60  # Height for the info bar
        canvas = np.zeros((height + info_height, width, 3), dtype=np.uint8)
        canvas[info_height:, :, :] = img
        
        # Add game mode and score information
        cv2.putText(
            canvas,
            f"Game: {game_info['game_name']}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        # Add player information
        cv2.putText(
            canvas,
            f"P1: {game_info['scores'].get('first_0', 0)}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )
        
        cv2.putText(
            canvas,
            f"P2: {game_info['scores'].get('second_0', 0)}",
            (width - 80, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )
        
        img = canvas
    
    # Use cv2 for faster encoding with compression
    success, encoded_img = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        logger.error("Failed to encode observation")
        return None
    return encoded_img.tobytes()

# Function to map key strings to actions
def key_to_action(key_state, env_name, player):
    """Map keyboard state to appropriate actions for the current environment."""
    if env_name == "pong_v3":
        # Pong actions: NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE
        if key_state.get("ArrowUp", False) or key_state.get("Key.up", False):
            return 2  # RIGHT (UP)
        elif key_state.get("ArrowDown", False) or key_state.get("Key.down", False):
            return 3  # LEFT (DOWN)
        elif key_state.get(" ", False) or key_state.get("Key.space", False):
            return 1  # FIRE
        else:
            return 0  # NOOP
    
    elif env_name == "space_invaders_v2":
        # Space Invaders actions: NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE
        if (key_state.get("ArrowRight", False) or key_state.get("Key.right", False)) and \
           (key_state.get(" ", False) or key_state.get("Key.space", False)):
            return 4  # RIGHTFIRE
        elif (key_state.get("ArrowLeft", False) or key_state.get("Key.left", False)) and \
             (key_state.get(" ", False) or key_state.get("Key.space", False)):
            return 5  # LEFTFIRE
        elif key_state.get("ArrowRight", False) or key_state.get("Key.right", False):
            return 2  # RIGHT
        elif key_state.get("ArrowLeft", False) or key_state.get("Key.left", False):
            return 3  # LEFT
        elif key_state.get(" ", False) or key_state.get("Key.space", False):
            return 1  # FIRE
        else:
            return 0  # NOOP
    
    elif env_name == "joust_v3":
        # Joust actions vary by player
        # Typically: NOOP, FIRE, UP, RIGHT, LEFT, RIGHTFIRE, UPFIRE, LEFTFIRE
        if (key_state.get("ArrowRight", False) or key_state.get("Key.right", False)) and \
           (key_state.get(" ", False) or key_state.get("Key.space", False)):
            return 6  # RIGHTFIRE
        elif (key_state.get("ArrowUp", False) or key_state.get("Key.up", False)) and \
             (key_state.get(" ", False) or key_state.get("Key.space", False)):
            return 7  # UPFIRE
        elif (key_state.get("ArrowLeft", False) or key_state.get("Key.left", False)) and \
             (key_state.get(" ", False) or key_state.get("Key.space", False)):
            return 8  # LEFTFIRE
        elif key_state.get("ArrowRight", False) or key_state.get("Key.right", False):
            return 3  # RIGHT
        elif key_state.get("ArrowUp", False) or key_state.get("Key.up", False):
            return 2  # UP
        elif key_state.get("ArrowLeft", False) or key_state.get("Key.left", False):
            return 4  # LEFT
        elif key_state.get(" ", False) or key_state.get("Key.space", False):
            return 1  # FIRE
        else:
            return 0  # NOOP
    
    # Default to NOOP for unknown environments
    return 0

class GameRoom:
    """A game room that manages a single game instance with two players."""
    def __init__(self, room_id, game_name="pong_v3"):
        self.room_id = room_id
        self.game_name = game_name
        self.env = None
        self.clients = {}
        self.running = False
        self.fps = 30
        self.frame_time = 1.0 / self.fps
        self.game_speed = 0.7
        self.scores = {}
        self.agents = []
        self.frame_times = deque(maxlen=100)
        self.task = None
        
    def start(self):
        """Initialize and start the game."""
        if self.env is None:
            self.env = self.create_env(self.game_name)
        
        self.running = True
        self.task = asyncio.create_task(self.game_loop())
        logger.info(f"Game room {self.room_id} started: {self.game_name}")
        
    def stop(self):
        """Stop the game."""
        self.running = False
        if self.task:
            self.task.cancel()
        if self.env:
            self.env.close()
            self.env = None
        logger.info(f"Game room {self.room_id} stopped")
        
    def create_env(self, game_name):
        """Create the appropriate environment based on the game name."""
        try:
            if game_name == "pong_v3":
                env = pong_v3.env()
            elif game_name == "space_invaders_v2":
                env = space_invaders_v2.env()
            elif game_name == "joust_v3":
                env = joust_v3.env()
            else:
                # Default to pong if game not recognized
                logger.warning(f"Unknown game: {game_name}, defaulting to pong_v3")
                env = pong_v3.env()
                game_name = "pong_v3"
            
            env.reset()
            self.game_name = game_name
            self.agents = env.agents.copy()  # Make a copy to prevent issues
            self.scores = {agent: 0 for agent in self.agents}
            logger.info(f"Created environment: {game_name} with agents: {self.agents}")
            return env
        except Exception as e:
            logger.error(f"Error creating environment: {e}")
            # Fallback to pong
            logger.info("Falling back to pong_v3")
            env = pong_v3.env()
            env.reset()
            self.game_name = "pong_v3"
            self.agents = env.agents.copy()  # Make a copy
            self.scores = {agent: 0 for agent in self.agents}
            return env
    
    def add_client(self, client_id, client_info):
        """Add a client to this game room."""
        # Find an available agent
        if len(self.clients) < 2 and self.agents:
            # Assign to the first available agent
            for agent in self.agents:
                if not any(c.get("agent") == agent for c in self.clients.values()):
                    client_info["agent"] = agent
                    client_info["room_id"] = self.room_id
                    self.clients[client_id] = client_info
                    logger.info(f"Added client {client_id} to room {self.room_id} as agent {agent}")
                    return True
                    
        # Room is full or no agents available
        logger.error(f"Could not add client {client_id} to room {self.room_id}. Agents: {self.agents}, Clients: {len(self.clients)}")
        return False
    
    def remove_client(self, client_id):
        """Remove a client from this game room."""
        if client_id in self.clients:
            logger.info(f"Removing client {client_id} from room {self.room_id}")
            del self.clients[client_id]
            # Stop the game if no clients left
            if not self.clients:
                self.stop()
            return True
        return False
    
    async def send_json(self, websocket, data):
        """Send JSON data with numpy handling."""
        try:
            json_str = json.dumps(data, cls=NumpyJSONEncoder)
            await websocket.send(json_str)
            return True
        except Exception as e:
            logger.error(f"Error sending JSON: {e}")
            return False
    
    async def game_loop(self):
        """Main game loop using PettingZoo environment."""
        prev_time = time.time()
        
        while self.running and self.clients:
            loop_start = time.time()
            
            # Calculate delta time since last frame
            delta = loop_start - prev_time
            prev_time = loop_start
            
            # Reset environment if needed
            if not hasattr(self.env, 'agents') or not self.env.agents:
                self.env.reset()
                await asyncio.sleep(0.1)  # Short delay after reset
                continue
            
            # Get the current agent
            agent = self.env.agent_selection
            
            # Find the client controlling this agent
            controlling_client = None
            for client_id, client in self.clients.items():
                if client["agent"] == agent:
                    controlling_client = client
                    break
            
            # Determine action for the current agent
            action = 0  # Default to NOOP
            
            if controlling_client:
                # Use the client's key state to determine action
                action = key_to_action(
                    controlling_client["key_state"], 
                    self.game_name,
                    agent
                )
                controlling_client["last_action"] = action
            
            try:
                # Step the environment with the chosen action
                self.env.step(action)
                
                # Update scores based on rewards (specific to each environment)
                if hasattr(self.env, 'rewards') and agent in self.env.rewards:
                    reward = self.env.rewards[agent]
                    if reward != 0:
                        self.scores[agent] = self.scores.get(agent, 0) + reward
                        # Notify clients of score change
                        score_update = {
                            "type": "score_update",
                            "scores": self.scores,
                            "player_scored": agent,
                            "reward": reward
                        }
                        for client_id, client in self.clients.items():
                            try:
                                await self.send_json(client["websocket"], score_update)
                            except:
                                pass
            except Exception as e:
                logger.error(f"Error stepping environment: {e}")
                # Try to reset if there was an error
                try:
                    self.env.reset()
                except:
                    pass
            
            # Send updated state to all clients
            for client_id, client in list(self.clients.items()):
                await self.send_state_to_client(client_id)
            
            # Calculate how long the frame took to process
            frame_duration = time.time() - loop_start
            self.frame_times.append(frame_duration)
            
            # Sleep to maintain target frame rate
            sleep_time = max(0, self.frame_time - frame_duration)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
    
    async def send_state_to_client(self, client_id):
        """Send game state to a specific client."""
        if client_id not in self.clients:
            return
        
        client = self.clients[client_id]
        
        # Get observation for the client's agent or the first agent if spectator
        obs = None
        if self.env is not None:
            if hasattr(self.env, 'agents') and self.env.agents:
                if client["agent"] in self.env.agents:
                    # Get observation for this specific agent
                    if hasattr(self.env, "observe"):
                        obs = self.env.observe(client["agent"])
                    else:
                        # Fallback to last observation
                        obs = self.env.last_observation
                else:
                    # Spectator sees the first agent's view
                    if hasattr(self.env, "observe"):
                        obs = self.env.observe(self.env.agents[0])
                    else:
                        obs = self.env.last_observation
        
        # Prepare game info for overlay
        game_info = {
            "game_name": self.game_name,
            "scores": self.scores,
            "current_agent": self.env.agent_selection if self.env and hasattr(self.env, 'agent_selection') else None,
            "player_id": client["player_id"],
            "agent": client["agent"],
            "room_id": self.room_id
        }
        
        # Encode observation with game info overlay
        encoded_data = encode_obs(obs, game_info)
        if encoded_data is None:
            return
        
        # Send to client
        try:
            await client["websocket"].send(encoded_data)
        except websockets.ConnectionClosed:
            logger.warning(f"Failed to send state to client {client_id}")
        except Exception as e:
            logger.error(f"Error sending state to client {client_id}: {e}")

class MatchmakingServer:
    """Server that manages client connections and matchmaking."""
    def __init__(self):
        self.waiting_clients = {}  # Clients waiting to be matched
        self.game_rooms = {}       # Active game rooms
        self.client_to_room = {}   # Mapping of client_id to room_id
        self.available_games = ["pong_v3", "space_invaders_v2", "joust_v3"]
        
    async def handle_client(self, websocket, path):
        """Handle a client connection."""
        # Generate a unique client ID
        client_id = str(uuid.uuid4())
        
        try:
            # Receive initial message
            init_message = await websocket.recv()
            
            try:
                data = json.loads(init_message)
                client_type = data.get("type", "HUMAN")
                player_id = data.get("player_id", None) or f"Player-{client_id[:6]}"
                requested_game = data.get("game", "pong_v3")
                client_name = data.get("name", f"Player-{client_id[:6]}")
            except json.JSONDecodeError:
                # Legacy client support
                client_type = "HUMAN"
                player_id = f"Player-{client_id[:6]}"
                requested_game = "pong_v3"
                client_name = f"Player-{client_id[:6]}"
                
            if requested_game not in self.available_games:
                requested_game = "pong_v3"
                
            # Store client information
            client_info = {
                "websocket": websocket,
                "type": client_type,
                "player_id": player_id,
                "name": client_name,
                "agent": None,  # Will be assigned when placed in a room
                "key_state": {},
                "last_ping": time.time(),
                "requested_game": requested_game
            }
            
            logger.info(f"Client {client_id} connected: {client_name} requesting {requested_game}")
            
            # Add to waiting list and try to match
            await self.add_to_waiting(client_id, client_info)
            
            # Send waiting message
            await websocket.send(json.dumps({
                "type": "waiting",
                "message": "Waiting for another player...",
                "client_id": client_id,
                "available_games": self.available_games
            }))
            
            # Handle client messages
            while True:
                message = await websocket.recv()
                
                try:
                    data = json.loads(message)
                    
                    # Handle ping-pong for latency measurement
                    if "ping" in data:
                        # Echo back the ping timestamp
                        await websocket.send(json.dumps({"pong": data["ping"]}))
                    
                    # Handle key state updates
                    if "keys" in data:
                        # Update key state in the right place based on where client is
                        if client_id in self.client_to_room:
                            room_id = self.client_to_room[client_id]
                            if room_id in self.game_rooms:
                                room = self.game_rooms[room_id]
                                if client_id in room.clients:
                                    room.clients[client_id]["key_state"] = data["keys"]
                                    room.clients[client_id]["last_ping"] = time.time()
                        elif client_id in self.waiting_clients:
                            self.waiting_clients[client_id]["key_state"] = data["keys"]
                            self.waiting_clients[client_id]["last_ping"] = time.time()
                    
                    # Handle game change request (only if waiting)
                    if "change_game" in data:
                        new_game = data["change_game"]
                        if new_game in self.available_games and client_id in self.waiting_clients:
                            self.waiting_clients[client_id]["requested_game"] = new_game
                            logger.info(f"Client {client_id} changed requested game to {new_game}")
                            await websocket.send(json.dumps({
                                "type": "game_changed",
                                "game": new_game
                            }))
                
                except json.JSONDecodeError:
                    # Ignore non-JSON messages
                    pass
        
        except websockets.ConnectionClosed:
            logger.info(f"Client {client_id} disconnected")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        finally:
            # Clean up client
            await self.remove_client(client_id)
    
    async def add_to_waiting(self, client_id, client_info):
        """Add client to waiting list and try to match with another player."""
        self.waiting_clients[client_id] = client_info
        
        # Try to find a matching player with compatible game preference
        requested_game = client_info["requested_game"]
        matched_client_id = None
        
        for waiting_id, waiting_info in list(self.waiting_clients.items()):
            if waiting_id != client_id and waiting_info["requested_game"] == requested_game:
                matched_client_id = waiting_id
                break
        
        # If match found, create a game room
        if matched_client_id:
            await self.create_game_room(client_id, matched_client_id, requested_game)
    
    async def create_game_room(self, client_id1, client_id2, game_name):
        """Create a new game room for matched players."""
        room_id = f"room-{str(uuid.uuid4())[:8]}"
        
        # Create the room
        room = GameRoom(room_id, game_name)
        
        # Initialize the game environment first
        try:
            room.env = room.create_env(game_name)
            if not room.agents:
                logger.error(f"No agents available in {game_name} environment")
                return False
        except Exception as e:
            logger.error(f"Failed to create environment for {game_name}: {e}")
            return False
            
        # Add to game rooms now that we know it works
        self.game_rooms[room_id] = room
        
        # Get clients from waiting list (safely)
        client1 = self.waiting_clients.get(client_id1)
        client2 = self.waiting_clients.get(client_id2)
        
        if not client1 or not client2:
            logger.error(f"One of the clients disappeared before room creation")
            if room_id in self.game_rooms:
                del self.game_rooms[room_id]
            return False
            
        # Remove clients from waiting list
        if client_id1 in self.waiting_clients:
            del self.waiting_clients[client_id1]
        if client_id2 in self.waiting_clients:
            del self.waiting_clients[client_id2]
            
        # Add clients to the room
        success1 = room.add_client(client_id1, client1)
        success2 = room.add_client(client_id2, client2)
        
        if success1 and success2:
            # Update mappings
            self.client_to_room[client_id1] = room_id
            self.client_to_room[client_id2] = room_id
            
            # Notify clients they've been matched
            match_message = {
                "type": "matched",
                "game": game_name,
                "room_id": room_id,
                "players": [
                    {"id": client_id1, "name": client1["name"], "agent": client1["agent"]},
                    {"id": client_id2, "name": client2["name"], "agent": client2["agent"]}
                ]
            }
            
            # Send match notification to both clients
            try:
                await client1["websocket"].send(json.dumps(match_message))
                await client2["websocket"].send(json.dumps(match_message))
            except Exception as e:
                logger.error(f"Error sending match notification: {e}")
                return False
            
            # Send game info to both clients
            for client_id, client in [(client_id1, client1), (client_id2, client2)]:
                try:
                    await client["websocket"].send(json.dumps({
                        "type": "game_info",
                        "game": game_name,
                        "player_id": client["player_id"],
                        "agent": client["agent"],
                        "is_agent": client["agent"] is not None,
                        "all_agents": room.agents,
                        "available_games": self.available_games,
                        "room_id": room_id
                    }))
                except Exception as e:
                    logger.error(f"Error sending game info: {e}")
                    return False
            
            # Start the game
            room.start()
            logger.info(f"Created game room {room_id} with {client1['name']} and {client2['name']} playing {game_name}")
            return True
        else:
            # Failed to add clients, clean up
            logger.error(f"Failed to add clients to room {room_id}. Agents available: {room.agents}")
            if room_id in self.game_rooms:
                self.game_rooms[room_id].stop()
                del self.game_rooms[room_id]
            
            # Put clients back in waiting queue
            self.waiting_clients[client_id1] = client1
            self.waiting_clients[client_id2] = client2
            
            logger.error(f"Failed to create game room for {client_id1} and {client_id2}")
            return False
    
    async def remove_client(self, client_id):
        """Remove client from server."""
        # Check if client is in a game room
        if client_id in self.client_to_room:
            room_id = self.client_to_room[client_id]
            if room_id in self.game_rooms:
                # Remove from room
                self.game_rooms[room_id].remove_client(client_id)
                
                # Check if room is empty
                if not self.game_rooms[room_id].clients:
                    self.game_rooms[room_id].stop()
                    del self.game_rooms[room_id]
                else:
                    # Notify remaining client
                    for remaining_id, remaining_client in self.game_rooms[room_id].clients.items():
                        try:
                            await remaining_client["websocket"].send(json.dumps({
                                "type": "opponent_left",
                                "message": "Your opponent has left the game."
                            }))
                        except:
                            pass
            
            # Remove mapping
            del self.client_to_room[client_id]
        
        # Check if client is waiting
        if client_id in self.waiting_clients:
            del self.waiting_clients[client_id]

# Start the WebSocket server
async def main():
    """Run the WebSocket server."""
    server = MatchmakingServer()
    
    host = "0.0.0.0"  # Listen on all network interfaces
    port = 8765
    
    async with websockets.serve(server.handle_client, host, port):
        logger.info(f"Matchmaking server started on ws://{host}:{port}")
        await asyncio.Future()  # Run indefinitely

if __name__ == "__main__":
    asyncio.run(main())