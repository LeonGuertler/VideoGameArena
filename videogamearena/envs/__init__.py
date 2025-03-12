""" Register all environments """

from videogamearena.envs.registration import register

# AirstrikerGenesis (single-player)
register(id="AirstrikerGenesis-v0", entry_point="videogamearena.envs.AirstrikerGenesis.env:AirstrikerGenesisEnv")

# MortalKombat (single-player)
register(id="MortalKombatII-v0", entry_point="videogamearena.envs.MortalKombatII.env:MortalKombatIIEnv", players=1)
register(id="MortalKombatII-2p-v0", entry_point="videogamearena.envs.MortalKombatII.env:MortalKombatIIEnv", players=2)
register(id="MortalKombatII-2p-v0-slow", entry_point="videogamearena.envs.MortalKombatII.env:MortalKombatIIEnv", speed_mode='slow', players=2)


# StreetFighter2 (single-player)
register(id="StreetFighterII-v0", entry_point="videogamearena.envs.StreetFighterII.env:StreetFighterIIEnv")

# SuperMarioBros (single-player)
register(id="SuperMarioBros-v0", entry_point="videogamearena.envs.SuperMarioBros.env:SuperMarioBrosEnv", speed_mode="human")
register(id="SuperMarioBros-v0-slow", entry_point="videogamearena.envs.SuperMarioBros.env:SuperMarioBrosEnv", speed_mode="slow")
register(id="SuperMarioBros-v0-super-slow", entry_point="videogamearena.envs.SuperMarioBros.env:SuperMarioBrosEnv", speed_mode="super-slow")

# Zelda (single-player)
register(id="Zelda-v0", entry_point="videogamearena.envs.Zelda.env:ZeldaEnv")
