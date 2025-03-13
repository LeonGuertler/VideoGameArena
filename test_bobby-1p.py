import time
import json
import gzip
import numpy as np
import cv2
import base64
import videogamearena as vga

agent = vga.agents.HumanAgent()

env = vga.make("AirstrikerGenesis-v0")

obs = env.reset()
done = False
raw_obs_data = []  # Store raw data first
frame_count = 0

while not done:
    env.render()
    action = agent(obs)
    print(action)
    obs, done, info = env.step(action)

    # Store raw observations
    raw_obs_data.append({
        "frame": frame_count,
        "visual": obs["visual"],  # Keep raw image for now
        "info": {key: obs[key] for key in obs if key != "visual"}  # Store metadata separately
    })

    frame_count += 1
    # if frame_count == 500:  # Limit to 3 frames for testing
    #     break

reward = env.close()
print(f"Reward: {reward}")
print("Info:", info)

# Now process and encode everything
compressed_obs_data = []

for frame_data in raw_obs_data:
    frame_id = frame_data["frame"]
    raw_image = frame_data["visual"]

    # Convert image to JPEG in-memory
    _, img_encoded = cv2.imencode(".jpg", cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    
    # Convert to base64
    img_base64 = base64.b64encode(img_encoded.tobytes()).decode("utf-8")

    # Convert metadata to JSON-friendly format
    metadata = {
        "frame": frame_id,
        "visual": img_base64,  # Embed image as base64 string
        "info": {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in frame_data["info"].items()}
    }
    compressed_obs_data.append(metadata)

# Save JSON metadata with gzip compression
with gzip.open("mario_obs.json.gz", "wt", encoding="utf-8") as f:
    json.dump(compressed_obs_data, f, indent=4)

print(f"Observations saved to mario_obs.json.gz with images embedded as Base64.")