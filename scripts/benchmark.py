# %%
from collections.abc import Callable
from dataclasses import dataclass
import logging
from pydantic import BaseModel
import math
import numpy as np
import hydra
from hydra.core.config_store import ConfigStore

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.benchmark import benchmark


# %%
import torch
from torch import Tensor
import torch.cuda.nvtx as nvtx
from jaxtyping import Float, Bool
from einops import einsum

import cs336_basics.model


@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
  d_k = K.shape[-1]
  with nvtx.range("computing attention scores"):
    attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

  if mask is not None:
    attention_scores = torch.where(mask, attention_scores, float("-inf"))

  with nvtx.range("computing softmax"):
    attention_weights = torch.softmax(attention_scores, dim=-1)  # Softmax over the key dimension

  with nvtx.range("final matmul"):
    output = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
  return output


cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

# %%


class LmConfig(BaseModel):
  vocab_size: int
  context_length: int
  d_model: int
  num_layers: int
  num_heads: int
  d_ff: int
  rope_theta: float

  batch_size: int

  def create_model_and_optimizer(self, device: str = "cuda") -> tuple[BasicsTransformerLM, AdamW]:
    model = BasicsTransformerLM(
        vocab_size=self.vocab_size,
        context_length=self.context_length,
        d_model=self.d_model,
        num_layers=self.num_layers,
        num_heads=self.num_heads,
        d_ff=self.d_ff,
        rope_theta=self.rope_theta,
    )
    model.to(device)
    optimizer = AdamW(model.parameters())
    return model, optimizer


def run_model_impl(config: LmConfig, backward: bool, device: str = "cuda") -> Callable:
  model, optimizer = config.create_model_and_optimizer(device=device)
  x = torch.randint(0, config.vocab_size, (config.batch_size,
                                           config.context_length), device=device)
  y = torch.randint(0, config.vocab_size, (config.batch_size,
                                           config.context_length), device=device)

  def model_run():
    y_pred = model(x)
    loss = cross_entropy(y_pred, y)
    if backward:
      loss.backward()
      # optimizer.step()
      # optimizer.zero_grad()
  return model_run


# %%
small_config = LmConfig(
    vocab_size=10_000,
    context_length=128,
    d_model=768,
    num_layers=12,
    num_heads=12,
    d_ff=3072,
    rope_theta=10000.0,
    batch_size=4,
)
medium_config = LmConfig(
    vocab_size=10_000,
    context_length=128,
    d_model=1024,
    num_layers=24,
    num_heads=16,
    d_ff=4096,
    rope_theta=10000.0,
    batch_size=4,
)
large_config = LmConfig(
    vocab_size=10_000,
    context_length=128,
    d_model=1280,
    num_layers=36,
    num_heads=20,
    d_ff=5120,
    rope_theta=10000.0,
    batch_size=4,
)
xl_config = LmConfig(
    vocab_size=10_000,
    context_length=128,
    d_model=1600,
    num_layers=48,
    num_heads=25,
    d_ff=6400,
    rope_theta=10000.0,
    batch_size=4,
)
_2_7B_config = LmConfig(
    vocab_size=10_000,
    context_length=128,
    d_model=2560,
    num_layers=32,
    num_heads=32,
    d_ff=10240,
    rope_theta=10000.0,
    batch_size=4,
)

dataname_to_config = {
    "small": small_config,
    "medium": medium_config,
    "large": large_config,
    "xl": xl_config,
    "2.7B": _2_7B_config,
}
# %%
config = LmConfig(
    vocab_size=10_000,
    context_length=256,
    d_model=768,
    num_layers=12,
    num_heads=12,
    d_ff=3072,
    rope_theta=10000.0,
    batch_size=16,
)


# %%


@dataclass
class BenchmarkConfig():
  data: str
  backward: bool
  warmup: int
  trial: int
  context_length: int


DEFAULT_BENCHMARK_CONFIG = BenchmarkConfig(
    data="small",
    warmup=5,
    trial=10,
    backward=True,
    context_length=128
)

cs = ConfigStore.instance()
cs.store(name="benchmark_config", node=DEFAULT_BENCHMARK_CONFIG)


@hydra.main(config_path=None, config_name="benchmark_config", version_base=None)
def run_benchmark(cfg: BenchmarkConfig):
  logging.info(f"Run Configuration: {cfg}")
  lm_config = dataname_to_config[cfg.data]
  lm_config.context_length = cfg.context_length
  logging.info(f"Model Configuration: {lm_config.model_dump()}")

  device = "cuda" if torch.cuda.is_available() else "cpu"
  run = run_model_impl(lm_config, backward=cfg.backward, device=device)
  desc = f"{cfg.data} backward={cfg.backward} [{cfg.warmup}, {cfg.trial}]"
  times = benchmark(desc, run, num_warmups=cfg.warmup, num_trials=cfg.trial)
  times = np.array(times)
  logging.info(desc)
  logging.info(f"times: {times}")
  logging.info(f"mean={times.mean():.4f}, std={times.std():.4f}")


# %%
if __name__ == "__main__":
  run_benchmark()
