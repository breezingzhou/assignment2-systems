# %%
from typing import Any, Iterable

import torch
import torch.distributed as dist
from torch.optim import Optimizer


class ShardedOptimizer(Optimizer):
  """Minimal ZeRO-style optimizer state sharding.

  Each rank owns optimizer state for a subset of parameters. Gradients are
  averaged across ranks; owners apply updates and broadcast weights so models
  remain in sync. Duplicate parameters (tied weights) are handled by only
  sharding unique parameter tensors.
  """

  def __init__(self, params: Iterable[torch.Tensor], optimizer_cls: type[Optimizer], **kwargs: Any):
    self.world_size = dist.get_world_size()
    self.rank = dist.get_rank()

    # Deduplicate to avoid double-optimizing tied/shared parameters.
    self.params: list[torch.Tensor] = []
    self.param_to_owner: dict[int, int] = {}
    for idx, param in enumerate(params):
      pid = id(param)
      if pid in self.param_to_owner:  # already seen tied weight
        continue
      self.param_to_owner[pid] = idx % self.world_size
      self.params.append(param)

    local_params = [p for p in self.params if self.param_to_owner[id(p)] == self.rank]
    self.local_optimizer = optimizer_cls(local_params, **kwargs)
    super().__init__(self.params, defaults=kwargs)

  def zero_grad(self, set_to_none: bool = True):
    for param in self.params:
      if param.grad is None:
        continue
      if set_to_none:
        param.grad = None
      else:
        param.grad.zero_()

  def step(self, closure=None, **kwargs):
    # Closure support mirrors Optimizer.step semantics.
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    # Aggregate gradients across ranks so the owning rank applies the same
    # update a single-process optimizer would.
    if self.world_size > 1:
      for param in self.params:
        if param.grad is None:
          continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(self.world_size)

    # Only the owner rank maintains optimizer state and performs the update.
    self.local_optimizer.step()

    # Broadcast updated parameters from the owning rank so that every replica stays identical.
    if self.world_size > 1:
      for param in self.params:
        owner = self.param_to_owner[id(param)]
        dist.broadcast(param.data, src=owner)

    return loss

  def add_param_group(self, param_group: dict[str, Any]):
    # Update ownership mapping and forward relevant parameters to the wrapped
    # optimizer. Use the current parameter count to determine ownership so that
    # sharding is stable under incremental additions.
    params = param_group.get("params")
    assert params is not None
    params = list(params)

    start_idx = len(self.params)
    new_params: list[torch.Tensor] = []
    for param in params:
      pid = id(param)
      if pid in self.param_to_owner:  # skip duplicates / tied weights already tracked
        continue
      owner = (start_idx + len(new_params)) % self.world_size
      self.param_to_owner[pid] = owner
      self.params.append(param)
      new_params.append(param)

    if not new_params:
      return

    local_params = [p for p in new_params if self.param_to_owner[id(p)] == self.rank]
    opts = {k: v for k, v in param_group.items() if k != "params"}
    if local_params:
      self.local_optimizer.add_param_group({"params": local_params, **opts})

    super().add_param_group({"params": new_params, **opts})
