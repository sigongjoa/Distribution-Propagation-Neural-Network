__version__ = "0.0.1"

from .core.config import DistConfig, DistMode, Preset, preset_config
from .core.distribution import BaseDistribution
from .dists.gaussian import GaussianDiag
from .layers.linear import DistLinear
from .layers.attention import DistSelfAttention
from .layers.blocks import DistTransformerBlock
