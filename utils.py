from typing import Optional, Callable, List

import torch

device = "cpu"
if torch.cuda.is_available():
   device = "cuda"

def _test_memory(
    func: Callable,
    _iter: int = 10,
    quantiles: Optional[List[float]] = None,
    return_mode="mean",
) -> float:
    assert return_mode in ["min", "max", "mean", "median"]
    total_mem = []

    for _ in range(_iter):
        getattr(torch, device).memory.reset_peak_memory_stats()
        func()
        # Convert to MiB
        mem = getattr(torch, device).max_memory_allocated() / 2**20
        total_mem.append(mem)

    total_mem = torch.tensor(total_mem, dtype=torch.float)
    if quantiles is not None:
        quantiles_data = torch.quantile(total_mem, torch.tensor(quantiles, dtype=torch.float)).tolist()
        if len(quantiles_data) == 1:
            quantiles_data = quantiles_data[0]
        return quantiles_data
    return getattr(torch, return_mode)(total_mem).item()

def _tensor_mem(a: torch.Tensor):
    # return memory allocated for a tensor in MiB
    return a.element_size() * a.numel() / 2**20