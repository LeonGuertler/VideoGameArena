import os
import asyncio
import websockets
import json
import base64
import cv2
import numpy as np
import videogamearena as vga
from collections import defaultdict
import http.server
import threading
import logging
from supabase import create_client, Client
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Supabase configuration
NEXT_PUBLIC_SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
NEXT_PUBLIC_SUPABASE_ANON_KEY = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
supabase: Client = create_client(NEXT_PUBLIC_SUPABASE_URL, NEXT_PUBLIC_SUPABASE_ANON_KEY)

PORT = int(os.getenv("WEBSOCKET_PORT", "8000"))
HEALTH_CHECK_PORT = int(os.getenv("HEALTH_CHECK_PORT", "8001"))

KEY_MAPPING_P1 = {
    "up": "u", "down": "d", "left": "l", "right": "r",
    "a": "a", "b": "b", "start": "s",
}
KEY_MAPPING_P2 = {
    "up": "U", "down": "D", "left": "L", "right": "R",
    "a": "A", "b": "B", "start": "S",
}

sessions = {}  # {game_id: {"env": env, "done": bool, "clients": {client_id: websocket}, "player_actions": {client_id: action}, "max_players": int, "player_mapping": {client_id: player_num}, "session_mapping": {client_id: session_id}}}
env_id_mapping = {}  # {env_name: id} - populated from Supabase
model_id_mapping = {}  # {model_name: id} - populated from Supabase

class HealthCheckHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        if '/health' not in args[0]:
            super().log_message(format, *args)

def start_health_check_server():
    server = http.server.HTTPServer(('0.0.0.0', HEALTH_CHECK_PORT), HealthCheckHandler)
    logger.info(f"Health check server started at http://0.0.0.0:{HEALTH_CHECK_PORT}/health")
    server.serve_forever()

async def fetch_environment_ids():
    """Fetch environment IDs from Supabase and populate env_id_mapping"""
    global env_id_mapping
    try:
        response = supabase.table('environments').select('id, env_name').execute()
        environments = response.data
        env_id_mapping = {env['env_name']: env['id'] for env in environments}
        logger.info(f"Loaded environment IDs: {env_id_mapping}")
    except Exception as e:
        logger.error(f"Failed to fetch environment IDs: {e}")
        raise

async def fetch_model_ids():
    """Fetch model IDs from Supabase and populate model_id_mapping"""
    global model_id_mapping
    try:
        response = supabase.table('models').select('id, model_name').execute()
        models = response.data
        model_id_mapping = {model['model_name']: model['id'] for model in models}
        logger.info(f"Loaded model IDs: {model_id_mapping}")
    except Exception as e:
        logger.error(f"Failed to fetch model IDs: {e}")
        raise

async def log_game_start(game_id: str, env_name: str):
    """Log the start of a game session to Supabase"""
    env_id = env_id_mapping.get(env_name)
    if env_id is None:
        logger.error(f"No environment ID found for {env_name}")
        return
    try:
        response = supabase.table('games').insert({
            'id': game_id,
            'environment_id': env_id,
            'status': 'active',
            'reason': 'game started'
        }).execute()
        logger.info(f"Logged game start for game ID: {game_id}")
    except Exception as e:
        logger.error(f"Failed to log game start for {game_id}: {e}")

async def update_game_status(game_id: str, status: str, reason: str):
    """Update the game status in Supabase"""
    try:
        supabase.table('games').update({
            'status': status,
            'reason': reason
        }).eq('id', game_id).execute()
        logger.info(f"Updated game status for {game_id} to {status}: {reason}")
    except Exception as e:
        logger.error(f"Failed to update game status for {game_id}: {e}")

async def fetch_human_id(session_id: str):
    """Fetch human_id from Supabase based on session_id (cookie_id)"""
    try:
        response = supabase.table('humans').select('id').eq('cookie_id', session_id).execute()
        if response.data and len(response.data) > 0:
            return response.data[0]['id']
        return None
    except Exception as e:
        logger.error(f"Failed to fetch human_id for session_id {session_id}: {e}")
        return None

async def determine_player_ids(session_id: str):
    """Determine human_id and model_id based on session_id format"""
    # Simple heuristic: UUID-like strings have hyphens and a specific length
    uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I)
    
    if uuid_pattern.match(session_id):
        # Treat as human session ID
        human_id = await fetch_human_id(session_id)
        model_id = 0  # Default for humans
        logger.info(f"Session ID {session_id} identified as human, human_id: {human_id}")
    else:
        # Treat as model name
        model_id = model_id_mapping.get(session_id)
        human_id = None
        logger.info(f"Session ID {session_id} identified as model, model_id: {model_id}")
    
    return human_id, model_id

async def log_player_game(game_id: str, env_name: str, client_id: str, player_num: int, reward: float, outcome: str, session_id: str):
    """Log player-specific game outcome to Supabase"""
    env_id = env_id_mapping.get(env_name)
    if env_id is None:
        logger.error(f"No environment ID found for {env_name}")
        return
    
    human_id, model_id = await determine_player_ids(session_id)
    
    try:
        response = supabase.table('player_games').insert({
            'game_id': game_id,
            'player_id': player_num,
            'reward': reward,
            'outcome': outcome,
            'env_id': env_id,
            'human_id': human_id,
            'model_id': model_id,
            'elo_change': None,  # Placeholder, calculate if needed
            'skill_change': None  # Placeholder, calculate if needed
        }).execute()
        logger.info(f"Logged player game for {client_id} (player {player_num}) in game {game_id}: {outcome}, human_id: {human_id}, model_id: {model_id}")
    except Exception as e:
        logger.error(f"Failed to log player game for {client_id} in game {game_id}: {e}")

async def broadcast_frame(clients, visual):
    """Broadcast frame to all connected clients"""
    if isinstance(visual, np.ndarray) and visual.ndim == 3 and visual.shape[2] in [3, 4]:
        img_bgr = cv2.cvtColor(visual, cv2.COLOR_RGB2BGR)
        _, img_encoded = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        img_base64 = base64.b64encode(img_encoded.tobytes()).decode("utf-8")
        frame_message = json.dumps({"type": "frame", "data": img_base64})
        for client_id, ws in list(clients.items()):
            try:
                await ws.send(frame_message)
            except websockets.ConnectionClosed:
                logger.warning(f"Failed to send frame to client {client_id}")

async def initialize_game_session(websocket, path: str) -> tuple:
    """Initialize game session and return necessary data or close connection on error."""
    try:
        _, env_name, session_id, game_id, client_id = path.split("/")
    except ValueError:
        logger.error(f"Invalid path format: {path}")
        await websocket.close(1000, "Invalid path format, expected /env_name/session_id/game_id/client_id")
        return None, None, None, None, None

    logger.info(f"New connection on port {PORT}: {env_name}/{session_id}/{game_id}/{client_id}")
    max_players = 2 if "2p" in env_name else 1

    if game_id not in sessions:
        try:
            env = vga.make(env_name, players=max_players)
            env.reset()
            sessions[game_id] = {
                "env": env,
                "done": False,
                "clients": {},
                "player_actions": {},
                "max_players": max_players,
                "player_mapping": {},
                "session_mapping": {}
            }
            await log_game_start(game_id, env_name)
        except Exception as e:
            await websocket.close(1000, f"Invalid game: {env_name}")
            logger.error(f"Error creating env: {e}")
            return None, None, None, None, None
    elif len(sessions[game_id]["clients"]) >= max_players:
        await websocket.close(1000, f"Game session full ({max_players} players)")
        return None, None, None, None, None
    elif sessions[game_id]["done"]:
        await websocket.close(1000, "Game session already completed")
        return None, None, None, None, None

    return env_name, session_id, game_id, client_id, max_players

async def setup_client_connection(game_id: str, client_id: str, websocket, session_id: str, max_players: int):
    """Set up client connection details in the session."""
    sessions[game_id]["clients"][client_id] = websocket
    player_num = len(sessions[game_id]["clients"]) - 1
    sessions[game_id]["player_mapping"][client_id] = player_num
    sessions[game_id]["session_mapping"][client_id] = session_id
    key_mapping = KEY_MAPPING_P1 if player_num == 0 else KEY_MAPPING_P2
    await broadcast_frame(sessions[game_id]["clients"], sessions[game_id]["env"].reset())
    return player_num, key_mapping

async def get_client_input(websocket, game_id: str, client_id: str, key_mapping: dict, frame_interval: float) -> None:
    """Receive and process client input."""
    try:
        message = await asyncio.wait_for(websocket.recv(), timeout=frame_interval)
        data = json.loads(message)
        if data["type"] == "input":
            keys = data.get("keys", [])
            action_str = "".join(f"[{key_mapping.get(key, 'n')}]" for key in keys)
            sessions[game_id]["player_actions"][client_id] = action_str
    except asyncio.TimeoutError:
        sessions[game_id]["player_actions"][client_id] = "[n]"

async def process_game_step(game_id: str, max_players: int) -> tuple:
    """Process one game step and return results."""
    if len(sessions[game_id]["clients"]) < max_players or sessions[game_id]["done"]:
        return None, None, None

    env = sessions[game_id]["env"]
    client_keys = list(sessions[game_id]["clients"].keys())
    
    if max_players == 1:
        action = sessions[game_id]["player_actions"].get(client_keys[0], "[n]")
    else:
        p1_action = sessions[game_id]["player_actions"].get(client_keys[0], "[n]")
        p2_action = sessions[game_id]["player_actions"].get(client_keys[1], "[N]")
        action = f"{p1_action} {p2_action}".strip()

    try:
        step_result, done, info = env.step(action)
        return step_result, done, info
    except Exception as e:
        logger.error(f"Error during game step for game {game_id}: {e}")
        sessions[game_id]["done"] = True
        await update_game_status(game_id, "error", f"Error during game step: {str(e)}")
        return None, True, None

async def handle_game_over(game_id: str, env_name: str, final_rewards: dict, max_players: int):
    """Handle game completion logic for all players."""
    sessions[game_id]["done"] = True
    logger.info(f"Game {game_id} marked as done")
    
    if max_players == 1:
        final_rewards = {0: final_rewards if isinstance(final_rewards, (int, float)) else 0}
    if not isinstance(final_rewards, dict):
        final_rewards = {0: 0, 1: 0}

    outcomes = determine_outcomes(final_rewards, max_players)
    game_over_messages = prepare_game_over_messages(game_id, final_rewards, outcomes)

    for c_id, ws in list(sessions[game_id]["clients"].items()):
        try:
            await ws.send(game_over_messages[c_id])
        except websockets.ConnectionClosed:
            logger.warning(f"Failed to send game over to client {c_id}")

    for c_id in sessions[game_id]["clients"]:
        player_num = sessions[game_id]["player_mapping"][c_id]
        session_id = sessions[game_id]["session_mapping"][c_id]
        await log_player_game(game_id, env_name, c_id, player_num, int(final_rewards.get(player_num, 0)), outcomes[player_num], session_id)

    await update_game_status(game_id, "finished", "game finished")
    for c_id, ws in list(sessions[game_id]["clients"].items()):
        try:
            await ws.close(1000, "Game Over")
        except websockets.ConnectionClosed:
            logger.warning(f"Client {c_id} already disconnected during cleanup")

def determine_outcomes(final_rewards: dict, max_players: int) -> dict:
    """Determine player outcomes based on rewards."""
    outcomes = {}
    if max_players == 2:
        if final_rewards.get(0, 0) > final_rewards.get(1, 0):
            outcomes[0] = "win"
            outcomes[1] = "loss"
        elif final_rewards.get(1, 0) > final_rewards.get(0, 0):
            outcomes[0] = "loss"
            outcomes[1] = "win"
        else:
            outcomes[0] = "draw"
            outcomes[1] = "draw"
    else:
        outcomes[0] = "win" if final_rewards.get(0, 0) > 0 else "loss" if final_rewards.get(0, 0) < 0 else "draw"
    return outcomes

def prepare_game_over_messages(game_id: str, final_rewards: dict, outcomes: dict) -> dict:
    """Prepare game over messages for all clients."""
    messages = {}
    for c_id in sessions[game_id]["clients"]:
        player_num = sessions[game_id]["player_mapping"][c_id]
        messages[c_id] = json.dumps({
            "type": "game_over",
            "info": {"reward": final_rewards.get(player_num, 0), "outcome": outcomes[player_num]}
        })
    return messages

async def handle_client(websocket):
    """Main client handler with simplified flow."""
    try:
        path = websocket.request.path
    except AttributeError:
        logger.error("Could not access request.path from websocket")
        await websocket.close(1000, "Unable to determine path")
        return

    env_name, session_id, game_id, client_id, max_players = await initialize_game_session(websocket, path)
    if not game_id:
        return

    player_num, key_mapping = await setup_client_connection(game_id, client_id, websocket, session_id, max_players)
    frame_interval = 1 / 60
    last_frame_time = asyncio.get_event_loop().time()

    try:
        while not sessions[game_id]["done"]:
            # 1. Get input
            await get_client_input(websocket, game_id, client_id, key_mapping, frame_interval)

            # 2. Process game step
            step_result, done, info = await process_game_step(game_id, max_players)
            if step_result:
                # 3. Send back image
                await broadcast_frame(sessions[game_id]["clients"], step_result["visual"])

            if done:
                final_rewards = sessions[game_id]["env"].close()
                logger.info(f"Game {game_id} Final Rewards: {final_rewards}")
                await handle_game_over(game_id, env_name, final_rewards, max_players)
                break

            # Frame rate control
            current_time = asyncio.get_event_loop().time()
            elapsed = current_time - last_frame_time
            sleep_time = max(0, frame_interval - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            last_frame_time = asyncio.get_event_loop().time()

    except websockets.ConnectionClosed:
        logger.info(f"Connection closed: {session_id}/{game_id}/{client_id}")
        if game_id in sessions and not sessions[game_id]["done"]:
            await update_game_status(game_id, "incomplete", "player disconnected")
    except Exception as e:
        logger.error(f"Unexpected error in handle_client for {client_id}: {e}")
    finally:
        if game_id in sessions:
            if client_id in sessions[game_id]["clients"]:
                del sessions[game_id]["clients"][client_id]
            if not sessions[game_id]["clients"]:
                if not sessions[game_id]["done"]:
                    try:
                        sessions[game_id]["env"].close()
                    except Exception as e:
                        logger.error(f"Error closing environment for game {game_id}: {e}")
                del sessions[game_id]
                logger.info(f"Game {game_id} closed and environment cleaned up")

async def main():
    await fetch_environment_ids()
    await fetch_model_ids()  # Load model IDs at startup
    
    health_thread = threading.Thread(target=start_health_check_server, daemon=True)
    health_thread.start()
    
    server = await websockets.serve(
        handle_client,
        "0.0.0.0",
        PORT,
        ping_interval=20,
        ping_timeout=60
    )
    logger.info(f"WebSocket server started at ws://0.0.0.0:{PORT}")
    await server.wait_closed()

if __name__ == "__main__":
    logger.info("Starting server.py")
    asyncio.run(main())