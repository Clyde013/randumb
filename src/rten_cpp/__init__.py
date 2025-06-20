# cpp extension called rten_cpp
import torch
from pathlib import Path
# This import will load the .so consisting of this file in this extension, so that the TORCH_LIBRARY static initializers are run,
# populating the torch.ops.rten_cpp namespace with the defined cpp implementations
from . import _C, ops