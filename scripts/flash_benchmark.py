# %%
import logging
from omegaconf import OmegaConf
from triton.testing import do_bench
from dataclasses import dataclass
import torch
import hydra
from hydra.core.config_store import ConfigStore
from cs336_systems.flashattn_triton import FlashAttnTriton
from torch.nn.modules import MultiheadAttention
from tests.test_attention import _attention_and_lse
# %%


@dataclass
class Config:
  n_seq: int
  d_model: int
  dtype: torch.dtype = torch.float32
  batch_size: int = 1
  device: str = "cuda"


@dataclass
class HydraConfig:
  n_seq: int
  d_model: int

  def into_config(self) -> Config:
    return Config(
        n_seq=self.n_seq,
        d_model=self.d_model,
    )


def make_attn_inputs(config: Config):
  # torch.random.manual_seed(0)
  batch_size = config.batch_size
  n_queries = config.n_seq
  n_keys = config.n_seq
  D = config.d_model
  device = config.device
  q = torch.randn(batch_size, n_queries, D, device=device, requires_grad=True)
  k = torch.randn(batch_size, n_keys, D, device=device, requires_grad=True)
  v = torch.randn(batch_size, n_keys, D, device=device, requires_grad=True)
  do = torch.randn(batch_size, n_queries, D, device=device)
  return q, k, v, do


def run_benchmark(config):
  q, k, v, do = make_attn_inputs(config)
  is_causal = True
  # forward
  triton_ms = do_bench(lambda: FlashAttnTriton.apply(
      q, k, v, is_causal))
  pytorch_ms = do_bench(lambda: _attention_and_lse(q, k, v, is_causal=is_causal))
  print("Triton FlashAttention forward time (ms):", triton_ms)
  print("PyTorch Attention forward time (ms):", pytorch_ms)

  # backward
  triton_o = FlashAttnTriton.apply(q, k, v, is_causal)
  pytorch_o = _attention_and_lse(q, k, v, is_causal=is_causal)[0]
  triton_ms_back = do_bench(lambda: triton_o.backward(do, retain_graph=True))  # type: ignore
  pytorch_ms_back = do_bench(lambda: pytorch_o.backward(do, retain_graph=True))
  print("Triton FlashAttention backward time (ms):", triton_ms_back)
  print("PyTorch Attention backward time (ms):", pytorch_ms_back)

  # forward + backward
  triton_ms_total = do_bench(lambda: FlashAttnTriton.apply(
      q, k, v, is_causal).backward(do))  # type: ignore
  pytorch_ms_total = do_bench(lambda: _attention_and_lse(
      q, k, v, is_causal=is_causal)[0].backward(do))
  print("Triton FlashAttention total time (ms):", triton_ms_total)
  print("PyTorch Attention total time (ms):", pytorch_ms_total)


# %%
DEFAULT_Hydra_CONFIG = HydraConfig(
    n_seq=128,
    d_model=16,
)
cs = ConfigStore.instance()
cs.store(name="benchmark_config", node=DEFAULT_Hydra_CONFIG)

# %%


@hydra.main(config_path=None, config_name="benchmark_config", version_base=None)
def main(in_cfg: OmegaConf):
  logging.info(f"Run Configuration: {in_cfg}")
  config = Config(**in_cfg)  # type: ignore
  logging.info(f"Benchmark Configuration: {config}")
  run_benchmark(config)


# %%
# python ./scripts/flash_benchmark.py -m d_model=16,32,64,128 n_seq=128,256,512,1024,2048,4096,8192,16384,32768,65536
if __name__ == "__main__":
  main()
