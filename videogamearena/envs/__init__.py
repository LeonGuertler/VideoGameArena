""" Register all environments """

from textarena.envs.registration import register


# SuperMarioBros (single-player)
register(id="SuperMarioBros-v0", entry_point="textarena.envs.SuperMarioBros.env:SuperMarioBrosEnv", speed_mode="human")
register(id="SuperMarioBros-v0-slow", entry_point="textarena.envs.SuperMarioBros.env:SuperMarioBrosEnv", speed_mode="slow")
register(id="SuperMarioBros-v0-super-slow", entry_point="textarena.envs.SuperMarioBros.env:SuperMarioBrosEnv", speed_mode="super-slow")

# Zelda (single-player)
register(id="Zelda-v0", entry_point="textarena.envs.Zelda.env:ZeldaEnv")
