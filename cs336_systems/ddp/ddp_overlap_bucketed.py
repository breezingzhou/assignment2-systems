# %%
import torch
import torch.distributed as dist
# %%


class Bucket:
  """Lightweight container for tracking bucket state during a step."""

  def __init__(self, params: list[torch.nn.Parameter]):
    self.params = params
    self.reset()

  def reset(self):
    self.ready_count = 0
    self.work: dist.Work | None = None
    self.flat: torch.Tensor | None = None

  def launch(self):
    grads: list[torch.Tensor] = []
    for param in self.params:
      if param.grad is None:
        param.grad = torch.zeros_like(param)
      grads.append(param.grad.contiguous().view(-1))

    flat = torch.cat(grads)
    work = dist.all_reduce(flat, op=dist.ReduceOp.SUM, async_op=True)
    self.flat = flat
    self.work = work


class DDPOverlapBucketed(torch.nn.Module):
  """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating buckets of gradients as they are ready
    in the backward pass.
  """

  def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
    super().__init__()
    self.module = module
    self.rank = dist.get_rank()
    self.world_size = dist.get_world_size()
    self.bucket_size_bytes = bucket_size_mb * 1024 * 1024
    self._pending_buckets: list[Bucket] = []
    self._buckets: list[Bucket] = []
    self._param_to_bucket: dict[int, Bucket] = {}
    if self.world_size == 1:
      return

    # Ensure all replicas start from identical parameters/buffers.
    for param in self._unique_parameters():
      dist.broadcast(param.data, src=0)
    for buffer in self.module.buffers():
      dist.broadcast(buffer, src=0)

    self._init_buckets()

    # Register per-parameter gradient hooks for overlapping all-reduce.
    for param in self._unique_parameters():
      if not param.requires_grad:
        continue

      bucket = self._param_to_bucket[id(param)]

      def _hook(param, bucket: Bucket = bucket):
        bucket.ready_count += 1
        if bucket.work is None and bucket.ready_count == len(bucket.params):
          self._launch_bucket(bucket)

      param.register_post_accumulate_grad_hook(_hook)

  def forward(self, *args, **kwargs):
    return self.module(*args, **kwargs)

  def start_gradient_synchronization(self):
    """Reset per-iteration bucket bookkeeping."""
    self._pending_buckets.clear()
    for bucket in self._buckets:
      bucket.reset()

  def finish_gradient_synchronization(self):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.
    """
    if self.world_size == 1:
      return

    for bucket in self._pending_buckets:
      assert bucket.work is not None and bucket.flat is not None
      bucket.work.wait()
      bucket.flat /= self.world_size

      offset = 0
      for param in bucket.params:
        assert param.grad is not None
        numel = param.grad.numel()
        param.grad.copy_(bucket.flat[offset: offset + numel].view(param.grad.shape))
        offset += numel

      bucket.reset()
    self._pending_buckets.clear()

  def _unique_parameters(self):
    """Yield each distinct Parameter once (handles tied weights)."""
    seen = set()
    for param in self.module.parameters():
      if id(param) in seen:
        continue
      seen.add(id(param))
      yield param

  def _init_buckets(self):
    current: list[torch.nn.Parameter] = []
    current_size = 0
    size_cap = self.bucket_size_bytes

    for param in self._unique_parameters():
      if not param.requires_grad:
        continue
      param_size = param.numel() * param.element_size()
      if current and current_size + param_size > size_cap:
        self._add_bucket(current)
        current = []
        current_size = 0

      current.append(param)
      current_size += param_size

    if current:
      self._add_bucket(current)

  def _add_bucket(self, params: list[torch.nn.Parameter]):
    bucket = Bucket(params)
    self._buckets.append(bucket)
    for p in params:
      self._param_to_bucket[id(p)] = bucket

  def _launch_bucket(self, bucket: Bucket):
    bucket.launch()
    self._pending_buckets.append(bucket)
