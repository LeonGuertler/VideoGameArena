from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional, Callable

from nes_py import NESEnv
import random


# Observation = ... 
# Info = Dict[str, Any]


# class Env(ABC):
#     """ TODO """
#     @abstractmethod
#     def reset(self, seed: Optional[int]=None):
#         """ TODO """
#         raise NotImplementedError

#     @abstractmethod
#     def setp(self, action) -> Tuple[bool, Observation, Info]:
#         """ TODO """
#         raise NotImplementedError

#     def close(self):
#         rewards = self.state.close()
#         return rewards 



# class NESEnvBaseClass(NESEnv):

#     def __init__(self, file_path):
#         rom_path = os.path.join(
#             "videogamearena", "envs", 
#             file_path
#         )
#         super(self).__init__(rom_path)

