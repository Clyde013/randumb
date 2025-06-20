# this file contains python wrapper functions over the torch.ops.rten_cpp namespace's cpp implementations,
# which is previously populated by the ._C import in __init__.py. This is just for convenience so that we
# don't have to call the full torch.ops namespace every time.
import torch
from torch import Tensor

__all__ = ["sq5_gen_2d", "materialize_fwd"]

def sq5_gen_2d(M: int, N: int, seed: int) -> Tensor:
    return torch.ops.rten_cpp.sq5_gen_2d.default(M, N, seed)

def materialize_fwd(coef: Tensor, seed: int, P: int, Q: int, stride_P: int, stride_Q: int) -> Tensor:
    return torch.ops.rten_cpp.materialize_fwd.default(coef, seed, P, Q, stride_P, stride_Q)
