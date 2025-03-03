import time 
from videogamearena.envs.SuperMarioBros.env import SuperMarioBrosEnv
from videogamearena.envs.Zelda.env import ZeldaEnv
from videogamearena.agents.basic_agents import OpenRouterAgent, HumanAgent

# agent = OpenRouterAgent(model_name="google/gemini-2.0-flash-001")
# agent = OpenRouterAgent(model_name="anthropic/claude-3.5-haiku")
# agent = OpenRouterAgent(model_name="meta-llama/llama-3.2-90b-vision-instruct")
# agent = OpenRouterAgent(model_name="meta-llama/llama-3.2-11b-vision-instruct")
agent = HumanAgent()

env = ZeldaEnv(speed_mode="human") #, rom_path="Legend of Zelda, The - A Link to the Past (USA).sfc")

obs = env.reset()
# input(obs)


done = False 

while not done:
    env.render()

    action = agent(obs)
    print(action)
    obs, done, info = env.step(action)

    # time.sleep(0.01)


reward = env.close()
print(f"Reward: {reward}")





