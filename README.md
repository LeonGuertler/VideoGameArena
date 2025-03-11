[![Website](https://img.shields.io/badge/videogamearena.ai-live%20site-blue)](https://videogamearena.ai)

# VideoGameArena
**VideoGameArena** is a flexible and extensible framework for training, evaluating, and benchmarking models and human players in video games. It follows an OpenAI Gym-style interface, making it straightforward to integrate with reinforcement learning agents, deep learning models, and human-controlled gameplay.

## Notes
- the same buttons should stay pressed until the next action is returned by the mode
- there should be a super slow mode
- properly check how long to sleep (and make sleep processing speed dependent) for consistency

## TODO
- set up env objects standardizing i/o s and wrapping the main envs 
- add universal control to agents (assess the environment keys)