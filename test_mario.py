# # import time 
# # from videogamearena.envs.SuperMarioBros.env import SuperMarioBrosEnv
# # from videogamearena.agents.basic_agents import OpenRouterAgent, HumanAgent

# # # agent = OpenRouterAgent(model_name="google/gemini-2.0-flash-001")
# # # agent = OpenRouterAgent(model_name="anthropic/claude-3.5-haiku")
# # # agent = OpenRouterAgent(model_name="meta-llama/llama-3.2-90b-vision-instruct")
# # # agent = OpenRouterAgent(model_name="meta-llama/llama-3.2-11b-vision-instruct")

# # agent = HumanAgent()
# # env = SuperMarioBrosEnv(speed_mode="human")

# # obs = env.reset()
# # # input(obs)


# # done = False 

# # full_obs = []

# # while not done:
# #     env.render()

# #     action = agent(obs)
# #     print(action)
# #     obs, done, info = env.step(action)

# #     full_obs.append(obs)

# #     # time.sleep(0.01)


# # reward = env.close()
# # print(f"Reward: {reward}")



# # import json 

# # # store full_obs as json




# # import time
# # import json
# # import numpy as np
# # from videogamearena.envs.SuperMarioBros.env import SuperMarioBrosEnv
# # from videogamearena.agents.basic_agents import OpenRouterAgent, HumanAgent

# # # agent = OpenRouterAgent(model_name="google/gemini-2.0-flash-001")
# # # agent = OpenRouterAgent(model_name="anthropic/claude-3.5-haiku")
# # # agent = OpenRouterAgent(model_name="meta-llama/llama-3.2-90b-vision-instruct")
# # # agent = OpenRouterAgent(model_name="meta-llama/llama-3.2-11b-vision-instruct")

# # agent = HumanAgent()
# # env = SuperMarioBrosEnv(speed_mode="human")

# # obs = env.reset()

# # done = False
# # full_obs = []

# # while not done:
# #     env.render()
    
# #     action = agent(obs)
# #     print(action)
# #     obs, done, info = env.step(action)

# #     # Convert ndarray to list before storing
# #     # obs["visual"] = obs["visual"].tolist()
# #     full_obs.append(obs) #["visual"].tolist() if isinstance(obs, np.ndarray) else obs)

# #     if len(full_obs) == 3:
# #         break

# # reward = env.close()
# # print(f"Reward: {reward}")

# # for i in range(len(full_obs)):
# #     full_obs[i]["visual"] = full_obs[i]["visual"].tolist()

# # # Save full_obs as a JSON file
# # with open("mario_obs.json", "w") as f:
# #     json.dump(full_obs, f)

# # print("Observations saved to mario_obs.json")


# # import time
# # import h5py
# # import numpy as np
# # from videogamearena.envs.SuperMarioBros.env import SuperMarioBrosEnv
# # from videogamearena.agents.basic_agents import OpenRouterAgent, HumanAgent

# # # agent = OpenRouterAgent(model_name="google/gemini-2.0-flash-001")
# # # agent = OpenRouterAgent(model_name="anthropic/claude-3.5-haiku")
# # # agent = OpenRouterAgent(model_name="meta-llama/llama-3.2-90b-vision-instruct")
# # # agent = OpenRouterAgent(model_name="meta-llama/llama-3.2-11b-vision-instruct")

# # agent = HumanAgent()
# # env = SuperMarioBrosEnv(speed_mode="human")

# # obs = env.reset()

# # done = False
# # full_obs = []

# # while not done:
# #     env.render()
    
# #     action = agent(obs)
# #     print(action)
# #     obs, done, info = env.step(action)

# #     obs["action"] = action
# #     full_obs.append(obs)

# #     # if len(full_obs) == 3:
# #     #     break

# # reward = env.close()
# # print(f"Reward: {reward}")

# # # Save observations using HDF5
# # with h5py.File("mario_obs.h5", "w") as hf:
# #     for i, obs in enumerate(full_obs):
# #         hf.create_dataset(f"frame_{i}", data=obs["visual"], compression="gzip")  # Store with compression

# # print("Observations saved to mario_obs.h5")


# import time
# import json
# import gzip
# import numpy as np
# from videogamearena.envs.SuperMarioBros.env import SuperMarioBrosEnv
# from videogamearena.agents.basic_agents import OpenRouterAgent, HumanAgent

# # agent = OpenRouterAgent(model_name="google/gemini-2.0-flash-001")
# # agent = OpenRouterAgent(model_name="anthropic/claude-3.5-haiku")
# # agent = OpenRouterAgent(model_name="meta-llama/llama-3.2-90b-vision-instruct")
# # agent = OpenRouterAgent(model_name="meta-llama/llama-3.2-11b-vision-instruct")

# agent = HumanAgent()
# env = SuperMarioBrosEnv(speed_mode="human")

# obs = env.reset()

# done = False
# full_obs = []

# # Function to recursively convert NumPy objects to serializable format
# def convert_to_serializable(obj):
#     if isinstance(obj, np.ndarray):
#         return obj.tolist()  # Convert NumPy arrays to lists
#     elif isinstance(obj, dict):
#         return {key: convert_to_serializable(value) for key, value in obj.items()}  # Recursively process dictionaries
#     elif isinstance(obj, list):
#         return [convert_to_serializable(item) for item in obj]  # Recursively process lists
#     elif isinstance(obj, (np.float32, np.float64)):
#         return float(obj)  # Convert NumPy floats
#     elif isinstance(obj, (np.int32, np.int64)):
#         return int(obj)  # Convert NumPy integers
#     else:
#         return obj  # Leave other types unchanged

# while not done:
#     env.render()
    
#     action = agent(obs)
#     print(action)
#     obs, done, info = env.step(action)

#     obs["action"] = action
#     full_obs.append(convert_to_serializable(obs))

#     if len(full_obs) == 250:  # Only store 3 frames for testing
#         break

# reward = env.close()
# print(f"Reward: {reward}")

# # Save JSON with gzip compression
# with gzip.open("mario_obs.json.gz", "wt", encoding="utf-8") as f:
#     json.dump(full_obs, f)

# print("Observations saved to mario_obs.json.gz")



import time
import json
import gzip
import numpy as np
import cv2
import base64
from io import BytesIO
from videogamearena.envs.SuperMarioBros.env import SuperMarioBrosEnv
from videogamearena.agents.basic_agents import OpenRouterAgent, HumanAgent

# agent = OpenRouterAgent(model_name="google/gemini-2.0-flash-001")
# agent = OpenRouterAgent(model_name="anthropic/claude-3.5-haiku")
# agent = OpenRouterAgent(model_name="meta-llama/llama-3.2-90b-vision-instruct")
# agent = OpenRouterAgent(model_name="meta-llama/llama-3.2-11b-vision-instruct")

agent = HumanAgent()
env = SuperMarioBrosEnv(speed_mode="human")

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
