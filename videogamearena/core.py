from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional, Callable

from nes_py import NESEnv
import random


Observation = Dict[str, Any] 
Info = Dict[str, Any]


class Env(ABC):
    """ TODO """
    @abstractmethod
    def reset(self, seed: Optional[int]=None) -> Observation:
        """ TODO """
        raise NotImplementedError

    @abstractmethod
    def step(self, action) -> Tuple[Observation, bool, Info]:
        """ TODO """
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class Wrapper(Env):
    """ Base class for environment wrappers. """

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, seed: Optional[int] = None) -> Observation:
        return self.env.reset(seed=seed)


    def step(self, action: str) -> Tuple[Observation, bool, Info]:
        return self.env.step(action=action)

    def close(self):
        return self.env.close()

class ActionWrapper(Wrapper):
    """ TODO """
    def step(self, action: str) -> Tuple[bool, Optional[Info]]:
        return self.env.step(action=self.action(action))

    def action(self, action: str) -> str:
        """
        Transforms the action.

        Args:
            action (str): The original action.

        Returns:
            str: The transformed action.
        """
        raise NotImplementedError

# class NESEnvBaseClass(NESEnv):

#     def __init__(self, file_path):
#         rom_path = os.path.join(
#             "videogamearena", "envs", 
#             file_path
#         )
#         super(self).__init__(rom_path)

