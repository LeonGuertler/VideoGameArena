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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Use an environment variable for dynamic WebSocket port (Default to 8000)
PORT = int(os.getenv("WEBSOCKET_PORT", "8000"))
# Health check port - using a different port (Default to 8001)
HEALTH_CHECK_PORT = int(os.getenv("HEALTH_CHECK_PORT", "8001"))

ENVIRONMENT_MAPPING = {
    "mario": "SuperMarioBros-v0",
    "mortal-kombat-ii": "MortalKombatII-v0",
    "zelda": "Zelda-v0"
}

KEY_MAPPING = {
    "up": 4, "down": 5, "left": 6, "right": 7,
    "a": 8, "b": 0, "start": 3,
}

ACTION_SIZE = 9
sessions = {}  # {session_id: {"env": env, "done": bool, "clients": set}}

# Simple HTTP server for health checks
class HealthCheckHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            # Simplified health check: if the server can respond, it’s healthy
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

async def handle_client(websocket):
    try:
        path = websocket.request.path
    except AttributeError:
        logger.error("Could not access request.path from websocket")
        await websocket.close(1000, "Unable to determine path")
        return

    try:
        _, slug, session_id, client_id = path.split("/")
        session_key = session_id
    except ValueError:
        await websocket.close(1000, "Invalid path format")
        return

    logger.info(f"New connection on port {PORT}: {slug}/{session_id}/{client_id}")

    env_name = ENVIRONMENT_MAPPING.get(slug.lower())
    if not env_name:
        await websocket.close(1000, f"Unknown game slug: {slug}")
        return

    if session_key in sessions:
        if sessions[session_key]["clients"]:
            await websocket.close(1000, "Session already in use by another client")
            logger.info(f"Rejected client {client_id} for session {session_id}: already in use")
            return
        if sessions[session_key]["done"]:
            sessions[session_key]["env"].close()
            del sessions[session_key]

    try:
        env = vga.make(env_name)
        obs = env.reset()
        sessions[session_key] = {
            "env": env,
            "done": False,
            "clients": {client_id}
        }
    except Exception as e:
        await websocket.close(1000, f"Invalid game: {slug}")
        logger.error(f"Error creating env: {e}")
        return

    env = sessions[session_key]["env"]
    done = sessions[session_key]["done"]

    async def send_frame(visual):
        if not done and isinstance(visual, np.ndarray):
            if visual.ndim == 3 and visual.shape[2] in [3, 4]:
                img_bgr = cv2.cvtColor(visual, cv2.COLOR_RGB2BGR)
                _, img_encoded = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                img_base64 = base64.b64encode(img_encoded.tobytes()).decode("utf-8")
                await websocket.send(json.dumps({"type": "frame", "data": img_base64}))

    await send_frame(obs)

    try:
        frame_interval = 1 / 60
        last_frame_time = asyncio.get_event_loop().time()

        while not done:
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=frame_interval)
                data = json.loads(message)
                if data["type"] == "input":
                    keys = data.get("keys", [])
                    action = [False] * ACTION_SIZE
                    for key in keys:
                        if key in KEY_MAPPING:
                            action[KEY_MAPPING[key]] = True
            except asyncio.TimeoutError:
                action = [False] * ACTION_SIZE

            step_result, done, info = env.step(action)
            visual = step_result["visual"]
            sessions[session_key]["done"] = done

            await send_frame(visual)

            if done:
                await websocket.send(json.dumps({
                    "type": "game_over",
                    "info": info
                }))
                await websocket.close(1000, "Game Over")
                break

            current_time = asyncio.get_event_loop().time()
            elapsed = current_time - last_frame_time
            sleep_time = max(0, frame_interval - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            last_frame_time = asyncio.get_event_loop().time()

    except websockets.ConnectionClosed:
        logger.info(f"Connection closed: {session_id}/{client_id}")
    finally:
        if session_key in sessions:
            sessions[session_key]["clients"].discard(client_id)
            if not sessions[session_key]["clients"]:
                sessions[session_key]["env"].close()
                del sessions[session_key]
                logger.info(f"Session {session_id} closed and environment cleaned up")

async def main():
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