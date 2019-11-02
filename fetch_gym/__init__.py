import distutils.version
import os
import sys
import warnings

from gym import error
from fetch_gym.utils import reraise
from gym.version import VERSION as __version__

from fetch_gym.core import Env, GoalEnv, Space, Wrapper, ObservationWrapper, ActionWrapper, RewardWrapper
from fetch_gym.envs import make, spec
from fetch_gym import wrappers, spaces, logger

def undo_logger_setup():
    warnings.warn("gym.undo_logger_setup is deprecated. gym no longer modifies the global logging configuration")

__all__ = ["Env", "Space", "Wrapper", "make", "spec", "wrappers"]
