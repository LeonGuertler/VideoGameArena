# import retro


# def main():
#     # env = retro.make(game="Airstriker-Genesis")
#     env = retro.make(game="Super_mario_brothers")
#     env.reset()
#     while True:
#         action = env.action_space.sample()
#         observation, reward, terminated, truncated, info = env.step(action)
#         env.render()
#         if terminated or truncated:
#             env.reset()
#     env.close()


# if __name__ == "__main__":
#     main()

# from nes_py.wrappers import JoypadSpace
# import gym_super_mario_bros
# from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
# env = gym_super_mario_bros.make('SuperMarioBros-v0')
# env = JoypadSpace(env, SIMPLE_MOVEMENT)

# done = True
# for step in range(5000):
#     if done:
#         state = env.reset()
#     state, reward, truncated, info = env.step(env.action_space.sample())
#     env.render()

# env.close()




# COMPLEX_MOVEMENT = [
#     ['NOOP'],
#     ['right'],
#     ['right', 'A'],
#     ['right', 'B'],
#     ['right', 'A', 'B'],
#     ['A'],
#     ['left'],
#     ['left', 'A'],
#     ['left', 'B'],
#     ['left', 'A', 'B'],
#     ['down'],
#     ['up'],
# ]

import time 
from videogamearena.envs.SuperMarioBros.env import SuperMarioBrosEnv
from videogamearena.agents.basic_agents import OpenRouterAgent, HumanAgent



# initialize agent
# agent = OpenRouterAgent(model_name="anthropic/claude-3-haiku")
agent = HumanAgent()

env = SuperMarioBrosEnv()

obs = env.reset()
# input(obs)


done = False 

while not done:
    env.render()

    action = agent(obs)

    obs, reward, done, info = env.step(action)

    # time.sleep(0.01)


reward = env.close()
print(f"Reward: {reward}")





