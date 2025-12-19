# %%

# %%
from collections.abc import Callable
import timeit

import torch


def benchmark(_description: str, run: Callable, num_warmups: int, num_trials: int):
  for _ in range(num_warmups):
    run()

  if torch.cuda.is_available():
    torch.cuda.synchronize()

  times: list[float] = []
  for trial in range(num_trials):
    start_time = timeit.default_timer()
    run()
    if torch.cuda.is_available():
      torch.cuda.synchronize()
    end_time = timeit.default_timer()
    times.append(end_time - start_time)
  return times
