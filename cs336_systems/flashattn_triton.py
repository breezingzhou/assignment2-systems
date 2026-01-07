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
    IS_CAUSAL: tl.constexpr,
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
  q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
  q_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

  # init o l m
  o = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
  l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
  m = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)

  for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
    # Load K and V tiles
    k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

    # Compute attention scores
    s = tl.dot(q, tl.trans(k))
    s = s * scale

    if IS_CAUSAL:
      k_offsets = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
      causal_mask = q_offsets[:, None] >= k_offsets[None, :]
      in_range = (q_offsets[:, None] < N_QUERIES) & (k_offsets[None, :] < N_KEYS)
      s = tl.where(causal_mask & in_range, s, -1.0e6)

    # compute logsumexp
    m_new = tl.maximum(m, tl.max(s, axis=1))
    p = tl.exp(s - m_new[:, None])
    alpha = tl.exp(m - m_new)
    l = alpha * l + tl.sum(p, axis=1)
    o = o * alpha[:, None] + tl.dot(p, v)
    m = m_new

    K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
    V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

  result_O = o / l[:, None]
  result_L = m + tl.log(l)
  tl.store(O_block_ptr, result_O, boundary_check=(0, 1))
  tl.store(L_block_ptr, result_L, boundary_check=(0,))


@triton.jit
def flash_bwd_D_kernel(
    O_ptr, dO_ptr, Delta_ptr,
    stride_ob, stride_oq, stride_od,
    stride_db, stride_dq,
    N_QUERIES, N_KEYS,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
):
  query_tile_index = tl.program_id(0)
  batch_index = tl.program_id(1)

  O_block_ptr = tl.make_block_ptr(
      O_ptr + batch_index * stride_ob,
      shape=(N_QUERIES, D),
      strides=(stride_oq, stride_od),
      offsets=(query_tile_index * Q_TILE_SIZE, 0),
      block_shape=(Q_TILE_SIZE, D),
      order=(1, 0),
  )
  dO_block_ptr = tl.make_block_ptr(
      dO_ptr + batch_index * stride_ob,
      shape=(N_QUERIES, D),
      strides=(stride_oq, stride_od),
      offsets=(query_tile_index * Q_TILE_SIZE, 0),
      block_shape=(Q_TILE_SIZE, D),
      order=(1, 0),
  )
  Delta_block_ptr = tl.make_block_ptr(
      Delta_ptr + batch_index * stride_db,
      shape=(N_QUERIES,),
      strides=(stride_dq,),
      offsets=(query_tile_index * Q_TILE_SIZE,),
      block_shape=(Q_TILE_SIZE,),
      order=(0,),
  )

  o = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
  do = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
  D = tl.sum((o * do), axis=-1)
  tl.store(Delta_block_ptr, D, boundary_check=(0,))


@triton.jit
def flash_bwd_kernel(
    dO_ptr, L_ptr, Delta_ptr,
    Q_ptr, K_ptr, V_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    IS_CAUSAL: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
  query_tile_index = tl.program_id(0)
  batch_index = tl.program_id(1)

  Q_block_ptr = tl.make_block_ptr(
      Q_ptr + batch_index * stride_qb,
      shape=(N_QUERIES, D),
      strides=(stride_qq, stride_qd),
      offsets=(query_tile_index * Q_TILE_SIZE, 0),
      block_shape=(Q_TILE_SIZE, D),
      order=(1, 0),
  )
  dQ_block_ptr = tl.make_block_ptr(
      dQ_ptr + batch_index * stride_qb,
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

  dO_block_ptr = tl.make_block_ptr(
      dO_ptr + batch_index * stride_qb,
      shape=(N_QUERIES, D),
      strides=(stride_qq, stride_qd),
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
  Delta_block_ptr = tl.make_block_ptr(
      Delta_ptr + batch_index * stride_lb,
      shape=(N_QUERIES,),
      strides=(stride_lq,),
      offsets=(query_tile_index * Q_TILE_SIZE,),
      block_shape=(Q_TILE_SIZE,),
      order=(0,),
  )

  q_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
  q_in_range = q_offsets < N_QUERIES

  q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
  l = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
  do = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
  delta = tl.load(Delta_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)

  dq = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

  for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
    k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

    s = tl.dot(q, tl.trans(k)) * scale

    k_offsets = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
    k_in_range = k_offsets < N_KEYS

    attn_mask = q_in_range[:, None] & k_in_range[None, :]
    if IS_CAUSAL:
      causal_mask = q_offsets[:, None] >= k_offsets[None, :]
      attn_mask = attn_mask & causal_mask
    s = tl.where(attn_mask, s, -1.0e6)

    p = tl.exp(s - l[:, None])

    # Compute dV
    dv = tl.dot(tl.trans(p), do)

    # Compute dp
    dp = tl.dot(do, tl.trans(v))

    # Compute ds
    # important!!! delta[:, None] to broadcast correctly
    ds = p * (dp - delta[:, None]) * scale

    # Compute dQ
    dq += tl.dot(ds, k)

    # Compute dK
    dk = tl.dot(tl.trans(ds), q)

    offs_d = tl.arange(0, D)

    dV_ptrs = (
        dV_ptr
        + batch_index * stride_vb
        + k_offsets[:, None] * stride_vk
        + offs_d[None, :] * stride_vd
    )
    dK_ptrs = (
        dK_ptr
        + batch_index * stride_kb
        + k_offsets[:, None] * stride_kk
        + offs_d[None, :] * stride_kd
    )

    mask_k = k_offsets[:, None] < N_KEYS

    tl.atomic_add(dV_ptrs, dv, mask=mask_k)
    tl.atomic_add(dK_ptrs, dk, mask=mask_k)

    K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
    V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))


  tl.store(dQ_block_ptr, dq, boundary_check=(0, 1))
# %%


class FlashAttnTriton(torch.autograd.Function):
  @staticmethod
  def forward(ctx, Q: Float[torch.Tensor, "batch_size queries d_k"], K: Float[torch.Tensor, "batch_size keys d_k"], V: Float[torch.Tensor, "batch_size keys d_v"], is_causal: Bool = False):
    for X in (Q, K, V):
      assert len(X.shape) == 3, "Input tensors to FlashAttention must be 3D."
      assert X.is_contiguous(), "Input tensors to FlashAttention must be contiguous."

    B = Q.shape[0]
    N_QUERIES = Q.shape[1]
    N_KEYS = K.shape[1]
    d = Q.shape[-1]
    scale = 1 / math.sqrt(d)

    O = torch.empty((B, N_QUERIES, d), dtype=Q.dtype, device=Q.device)
    # keep L in fp32 for numerical stability; tests only require it be saved
    L = torch.empty((B, N_QUERIES), dtype=torch.float32, device=Q.device)

    ctx.Q_TILE_SIZE = 16  # type: ignore
    ctx.K_TILE_SIZE = 16  # type: ignore

    grid = (triton.cdiv(N_QUERIES, ctx.Q_TILE_SIZE), B)
    flash_fwd_kernel[grid](  # type: ignore
        Q, K, V,
        O, L,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        L.stride(0), L.stride(1),
        N_QUERIES, N_KEYS,
        scale,
        IS_CAUSAL=is_causal,
        D=d,  # type: ignore[arg-type]
        Q_TILE_SIZE=ctx.Q_TILE_SIZE,  # type: ignore[arg-type]
        K_TILE_SIZE=ctx.K_TILE_SIZE,  # type: ignore[arg-type]
    )

    ctx.save_for_backward(Q, K, V, O, L)
    ctx.IS_CAUSAL = is_causal
    return O

  @staticmethod
  def backward(ctx, grad_output):
    (Q, K, V, O, L) = ctx.saved_tensors

    B = Q.shape[0]
    N_QUERIES = Q.shape[1]
    N_KEYS = K.shape[1]
    d = Q.shape[-1]
    scale = 1 / math.sqrt(d)

    dO = grad_output.contiguous()
    dQ = torch.zeros_like(Q, dtype=torch.float32, device=Q.device)
    dK = torch.zeros_like(K, dtype=torch.float32, device=K.device)
    dV = torch.zeros_like(V, dtype=torch.float32, device=V.device)
    Delta = torch.zeros_like(L, dtype=torch.float32, device=L.device)

    grid_delta = (triton.cdiv(N_QUERIES, ctx.Q_TILE_SIZE), B)
    flash_bwd_D_kernel[grid_delta](
        O, dO, Delta,
        O.stride(0), O.stride(1), O.stride(2),
        Delta.stride(0), Delta.stride(1),
        N_QUERIES, N_KEYS,
        D=d,  # type: ignore[arg-type]
        Q_TILE_SIZE=ctx.Q_TILE_SIZE,  # type: ignore[arg-type]
    )

    grid_main = (triton.cdiv(N_QUERIES, ctx.Q_TILE_SIZE), B)
    flash_bwd_kernel[grid_main](
        dO, L, Delta,
        Q, K, V,
        dQ, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        L.stride(0), L.stride(1),
        N_QUERIES, N_KEYS,
        scale,
        IS_CAUSAL=ctx.IS_CAUSAL,
        D=d,  # type: ignore[arg-type]
        Q_TILE_SIZE=ctx.Q_TILE_SIZE,  # type: ignore[arg-type]
        K_TILE_SIZE=ctx.K_TILE_SIZE,  # type: ignore[arg-type]
    )

    return dQ, dK, dV, None
