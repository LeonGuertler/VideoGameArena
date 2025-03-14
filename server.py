import os
import asyncio
import websockets
import json
import base64
import cv2
import numpy as np
import videogamearena as vga
from collections import defaultdict

# ✅ Use an environment variable for dynamic WebSocket port (Default to 8000)
PORT = int(os.getenv("WEBSOCKET_PORT", "8000"))

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

async def handle_client(websocket):
    try:
        path = websocket.request.path
    except AttributeError:
        print("Error: Could not access request.path from websocket")
        await websocket.close(1000, "Unable to determine path")
        return

    try:
        _, slug, session_id, client_id = path.split("/")
        session_key = session_id  # Use session_id as key
    except ValueError:
        await websocket.close(1000, "Invalid path format")
        return

    print(f"New connection on port {PORT}: {slug}/{session_id}/{client_id}")

    env_name = ENVIRONMENT_MAPPING.get(slug.lower())
    if not env_name:
        await websocket.close(1000, f"Unknown game slug: {slug}")
        return

    if session_key in sessions:
        if sessions[session_key]["done"]:
            sessions[session_key]["env"].close()
            del sessions[session_key]
        else:
            sessions[session_key]["clients"].add(client_id)
            print(f"Client {client_id} joined existing session {session_id}")
    else:
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
            print(f"Error creating env: {e}")
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

    await send_frame(obs if session_key not in sessions else env.render())

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
        print(f"Connection closed: {session_id}/{client_id}")
    finally:
        if session_key in sessions:
            sessions[session_key]["clients"].discard(client_id)
            if not sessions[session_key]["clients"]:
                sessions[session_key]["env"].close()
                del sessions[session_key]
                print(f"Session {session_id} closed and environment cleaned up")

async def main():
    server = await websockets.serve(
        handle_client,
        "0.0.0.0",
        PORT,  # ✅ Use dynamic port
        ping_interval=20,
        ping_timeout=60
    )
    print(f"WebSocket server started at ws://0.0.0.0:{PORT}")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())