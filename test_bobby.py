# import time
# import json
# import gzip
# import numpy as np
# import cv2
# import base64
# import videogamearena as vga

# agent = vga.agents.HumanAgent()

# env = vga.make("MortalKombatII-2p-v0")

# obs = env.reset()
# done = False
# raw_obs_data = []  # Store raw data first
# frame_count = 0

# while not done:
#     env.render()
#     action = agent(obs)
#     print(action)
#     obs, done, info = env.step(action)

#     # Store raw observations
#     raw_obs_data.append({
#         "frame": frame_count,
#         "visual": obs["visual"],  # Keep raw image for now
#         "info": {key: obs[key] for key in obs if key != "visual"}  # Store metadata separately
#     })

#     frame_count += 1
#     # if frame_count == 500:  # Limit to 3 frames for testing
#     #     break

# reward = env.close()
# print(f"Reward: {reward}")

# # Now process and encode everything
# compressed_obs_data = []

# for frame_data in raw_obs_data:
#     frame_id = frame_data["frame"]
#     raw_image = frame_data["visual"]

#     # Convert image to JPEG in-memory
#     _, img_encoded = cv2.imencode(".jpg", cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    
#     # Convert to base64
#     img_base64 = base64.b64encode(img_encoded.tobytes()).decode("utf-8")

#     # Convert metadata to JSON-friendly format
#     metadata = {
#         "frame": frame_id,
#         "visual": img_base64,  # Embed image as base64 string
#         "info": {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in frame_data["info"].items()}
#     }
#     compressed_obs_data.append(metadata)

# # Save JSON metadata with gzip compression
# with gzip.open("mario_obs.json.gz", "wt", encoding="utf-8") as f:
#     json.dump(compressed_obs_data, f, indent=4)

# print(f"Observations saved to mario_obs.json.gz with images embedded as Base64.")


import time
import json
import gzip
import numpy as np
import cv2
import base64
import videogamearena as vga
import re

# Create two human agents (both output lowercase actions)
agent1 = vga.agents.HumanAgent()  # Player 1
# agent2 = vga.agents.HumanAgent()  # Player 2

# agent1 = vga.agents.OpenRouterAgent("openai/gpt-4o-mini")
agent2 = vga.agents.OpenRouterAgent("anthropic/claude-3.5-haiku")

# Assuming MortalKombatIIEnv is registered as "MortalKombatII-2p-v0" in videogamearena
env = vga.make("MortalKombatII-2p-v0-slow")

obs = env.reset()
done = False
raw_obs_data = []  # Store raw data first
frame_count = 0

while not done:
    env.render()
    
    # Get actions from both players (both return lowercase initially)
    p1_action = agent1(obs)  # e.g., "[u] [a]"
    p2_action = agent2(obs)  # e.g., "[l] [x]"
    
    # Wrapper: Uppercase P2's actions
    p2_action_upper = ''.join(f"[{char.upper()}]" for char in re.findall(r'\[([^\]]*)\]', p2_action))
    
    # Combine actions, avoiding duplication of "[n]" if P2 has no action
    combined_action = p1_action + " " + p2_action_upper if p2_action_upper != "[N]" else p1_action
    
    print(f"P1 Action: {p1_action}")
    print(f"P2 Action (uppercased): {p2_action_upper}")
    print(f"Combined Action: {combined_action}")
    
    obs, done, info = env.step(combined_action)

    # Store raw observations
    raw_obs_data.append({
        "frame": frame_count,
        "visual": obs["visual"],
        "info": info # {key: obs[key] for key in obs if key != "visual"}
    })

    frame_count += 1
    # Uncomment to limit frames for testing
    # if frame_count == 500:
    #     break

reward = env.close()
print(f"Reward: {reward}")

# Process and encode observations
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
        "visual": img_base64,
        "info": {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in frame_data["info"].items()}
    }
    compressed_obs_data.append(metadata)

# Save JSON metadata with gzip compression
with gzip.open("mkii_obs.json.gz", "wt", encoding="utf-8") as f:
    json.dump(compressed_obs_data, f, indent=4)

print(f"Observations saved to mkii_obs.json.gz with images embedded as Base64.")