import time 
from videogamearena.envs.SuperMarioBros.env import SuperMarioBrosEnv
from videogamearena.agents.basic_agents import OpenRouterAgent, HumanAgent

# import nes_py 
# env = SuperMarioBrosEnv()
# nes_py.play_human(env=env)

# exit()



# initialize agent
# agent = HumanAgent()
agent = OpenRouterAgent(model_name="anthropic/claude-3-haiku")

env = SuperMarioBrosEnv()

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





