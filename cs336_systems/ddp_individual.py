# %%
import torch
import torch.distributed as dist
# %%


class DDPIndividualParameters(torch.nn.Module):
  """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating gradients as they are ready
    in the backward pass. The gradient for each parameter tensor
    is individually communicated.
  """

  def __init__(self, module: torch.nn.Module):
    super().__init__()
    self.module = module
    self.rank = dist.get_rank()
    self.world_size = dist.get_world_size()
    self._pending_works = []
    if self.world_size == 1:
      return

    # Ensure all replicas start from identical parameters/buffers.
    for param in self._unique_parameters():
      dist.broadcast(param.data, src=0)
    for buffer in self.module.buffers():
      dist.broadcast(buffer, src=0)

    # Register per-parameter gradient hooks for overlapping all-reduce.
    for param in self._unique_parameters():
      if not param.requires_grad:
        continue

      def _hook(param):
        if self.world_size == 1:
          return
        work = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
        self._pending_works.append((work, param.grad))

      param.register_post_accumulate_grad_hook(_hook)

  def forward(self, *args, **kwargs):
    return self.module(*args, **kwargs)

  def finish_gradient_synchronization(self):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.
    """
    if self.world_size == 1:
      return

    for work, grad in self._pending_works:
      work.wait()
      grad /= self.world_size

    self._pending_works.clear()

  def _unique_parameters(self):
    """Yield each distinct Parameter once (handles tied weights)."""
    seen = set()
    for param in self.module.parameters():
      if id(param) in seen:
        continue
      seen.add(id(param))
      yield param
