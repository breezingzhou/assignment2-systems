# %%
import torch
import torch.distributed as dist

# %%


class NaiveDDP(torch.nn.Module):
  """
  Returns a torch.nn.Module container that handles
  parameter broadcasting and gradient synchronization for
  distributed data parallel training.

  This container should synchronize gradients for all
  parameters together in a single all-reduce operation.

  """

  def __init__(self, module: torch.nn.Module):
    super().__init__()
    self.module = module
    self.rank = dist.get_rank()
    self.world_size = dist.get_world_size()

    if self.world_size == 1:
      return

    # Ensure all replicas start from identical parameters/buffers.
    for param in self.module.parameters():
      dist.broadcast(param.data, src=0)
    for buffer in self.module.buffers():
      dist.broadcast(buffer, src=0)

  def forward(self, *args, **kwargs):
    return self.module(*args, **kwargs)

  def sync_gradient(self):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.
    """
    if self.world_size == 1:
      return

    grads: list[torch.Tensor] = []
    shapes: list[torch.Size] = []
    params: list[torch.nn.Parameter] = []

    for param in self.module.parameters():
      if not param.requires_grad or param.grad is None:
        continue
      grads.append(param.grad.view(-1))
      shapes.append(param.grad.shape)
      params.append(param)

    if not grads:
      return

    flat = torch.cat([g.contiguous() for g in grads])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat /= self.world_size

    offset = 0
    for param, shape in zip(params, shapes):
      assert param.grad is not None
      numel = param.grad.numel()
      param.grad.copy_(flat[offset: offset + numel].view(shape))
      offset += numel


# %%
