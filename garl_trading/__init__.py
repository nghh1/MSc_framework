"""Group-Agent Reinforcement Learning trading research framework.

GARL follows the approach proposed by Wu and Zeng in their University of Manchester paper.
"""

from .config import FrameworkConfig, load_config

__all__ = ["FrameworkConfig", "load_config"]
__version__ = "0.1.0"
