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


def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                print('Tensor with grad found:', tensor)
                print(' - gradient:', tensor.grad)
                print()
            except AttributeError as e:
                getBack(n[0])