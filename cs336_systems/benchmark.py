# %%
from collections.abc import Callable
import timeit
import logging
import torch

# %%
logger = logging.getLogger(__name__)

# %%


def benchmark(_description: str, run: Callable, num_warmups: int, num_trials: int):
  logger.info(f"Benchmarking: {_description}")
  for warmup in range(num_warmups):
    logger.debug(f"Warmup {warmup + 1} / {num_warmups}")
    run()

  if torch.cuda.is_available():
    torch.cuda.synchronize()

  times: list[float] = []
  for trial in range(num_trials):
    logger.debug(f"Trial {trial + 1} / {num_trials}")
    start_time = timeit.default_timer()
    run()
    if torch.cuda.is_available():
      torch.cuda.synchronize()
    end_time = timeit.default_timer()
    times.append(end_time - start_time)
  return times
