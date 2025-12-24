# %%
import math
from einops import rearrange, einsum
import torch
from torch import Tensor
from jaxtyping import Float, Bool
import triton
import triton.language as tl

# %%


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):

  # Program indices
  query_tile_index = tl.program_id(0)
  batch_index = tl.program_id(1)
  # Offset each pointer with the corresponding batch index
  # multiplied with the batch stride for each tensor
  Q_block_ptr = tl.make_block_ptr(
      Q_ptr + batch_index * stride_qb,
      shape=(N_QUERIES, D),
      strides=(stride_qq, stride_qd),
      offsets=(query_tile_index * Q_TILE_SIZE, 0),
      block_shape=(Q_TILE_SIZE, D),
      order=(1, 0),
  )
  K_block_ptr = tl.make_block_ptr(
      K_ptr + batch_index * stride_kb,
      shape=(N_KEYS, D),
      strides=(stride_kk, stride_kd),
      offsets=(0, 0),
      block_shape=(K_TILE_SIZE, D),
      order=(1, 0),
  )
  V_block_ptr = tl.make_block_ptr(
      V_ptr + batch_index * stride_vb,
      shape=(N_KEYS, D),
      strides=(stride_vk, stride_vd),
      offsets=(0, 0),
      block_shape=(K_TILE_SIZE, D),
      order=(1, 0),
  )
  O_block_ptr = tl.make_block_ptr(
      O_ptr + batch_index * stride_ob,
      shape=(N_QUERIES, D),
      strides=(stride_oq, stride_od),
      offsets=(query_tile_index * Q_TILE_SIZE, 0),
      block_shape=(Q_TILE_SIZE, D),
      order=(1, 0),
  )
  L_block_ptr = tl.make_block_ptr(
      L_ptr + batch_index * stride_lb,
      shape=(N_QUERIES,),
      strides=(stride_lq,),
      offsets=(query_tile_index * Q_TILE_SIZE,),
      block_shape=(Q_TILE_SIZE,),
      order=(0,),
  )
  # Load Q tile
  q = tl.load(Q_block_ptr, boundary_check=(0, 0), padding_option="zero")
  # init o l m
  o = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
  l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
  m = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)

  for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
    # Load K and V tiles
    k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

    # Compute attention scores
    s = tl.dot(q, tl.trans(k))  # (Q_TILE_SIZE,D) x (K_TILE_SIZE,D) -> (Q_TILE_SIZE,K_TILE_SIZE)
    s = s * scale

    # compute logsumexp
    m_new = tl.maximum(m, tl.max(s, axis=1))
    p = tl.exp(s - m_new[:, None])
    l_new = tl.exp(m - m_new) * l + tl.sum(p, axis=1)
    o_new = tl.dot(tl.exp(m - m_new)[:, None], o) + tl.dot(p, v)
    o, l, m = o_new, l_new, m_new
    K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
    V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

  result_O = tl.dot(1 / l, o)
  result_L = m + tl.log(l)
  tl.store(O_block_ptr, result_O, boundary_check=(0, 1))
  tl.store(L_block_ptr, result_L, boundary_check=(0,))

# %%


class FlashAttnTriton(torch.autograd.Function):
  @staticmethod
  def forward(ctx, Q: Float[torch.Tensor, "... queries d_k"], K: Float[torch.Tensor, "... keys d_k"], V: Float[torch.Tensor, "... keys d_v"], is_causal: Bool = False):
    output_dims = Q.shape
    N_BATCHES = Q.shape[0]
    N_QUERIES = Q.shape[-2]
    N_KEYS = K.shape[-2]
    d = Q.shape[-1]
    scale = 1 / math.sqrt(d)

    Q, K, V = (
        rearrange(X, "batch_size (...) d -> (...) batch_size d")
        for X in (Q, K, V)
    )
    O = torch.empty_like(Q)
    L = torch.empty(Q.shape[:-1], dtype=Q.dtype, device=Q.device)

    ctx.Q_TILE_SIZE = 16  # type: ignore
    ctx.K_TILE_SIZE = 16  # type: ignore
    ctx.B_TILE_SIZE = 4

    flash_fwd_kernel[(triton.cdiv(N_QUERIES, ctx.Q_TILE_SIZE), triton.cdiv(N_BATCHES, ctx.B_TILE_SIZE))](
        Q, K, V,
        O, L,
        Q.stride(1), Q.stride(0), Q.stride(2),
        K.stride(1), K.stride(0), K.stride(2),
        V.stride(1), V.stride(0), V.stride(2),
        O.stride(1), O.stride(0), O.stride(2),
        L.stride(1), L.stride(0),
        N_QUERIES, N_KEYS,
        scale,
        D=d,
        Q_TILE_SIZE=ctx.Q_TILE_SIZE,
        K_TILE_SIZE=ctx.K_TILE_SIZE,
    )
    ctx.save_for_backward(Q, K, V, L)
    result_O = rearrange(O, "(...) batch_size d -> batch_size (...) d", batch_size=N_BATCHES)
    result_L = rearrange(L, "(...) batch_size -> batch_size (...)", batch_size=N_BATCHES)
    return result_O, result_L
