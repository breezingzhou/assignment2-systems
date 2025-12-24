# %%
import math
from einops import rearrange, einsum
import torch
from torch import Tensor
from jaxtyping import Float, Bool

# %%


class FlashAttnPytorch(torch.autograd.Function):
  @staticmethod
  def forward(ctx, Q: Float[torch.Tensor, "... queries d_k"], K: Float[torch.Tensor, "... keys d_k"], V: Float[torch.Tensor, "... keys d_v"], is_causal: Bool = False):
    q_shape = Q.shape
    k_shape = K.shape
    v_shape = V.shape
    Q, K, V = (
        rearrange(X, "... seq d -> (...) seq d")
        for X in (Q, K, V)
    )
    for X in (Q, K, V):
      assert len(X.shape) == 3, "Input tensors to FlashAttention must be 3D."
      assert X.is_contiguous(), "Input tensors to FlashAttention must be contiguous."

    d = q_shape[-1]
    scale = 1 / math.sqrt(d)

    Q_TILE_SIZE = 16
    K_TILE_SIZE = 16

    result_O = []
    result_L = []
    for q_start in range(0, Q.shape[1], Q_TILE_SIZE):
      q_end = min(q_start + Q_TILE_SIZE, Q.shape[1])
      q = Q[:, q_start:q_end]  # type: Float[Tensor, "BATCH_SIZE Q_TILE_SIZE d"]

      # init O_i, L_i, M_i
      # type: Float[Tensor, "BATCH_SIZE Q_TILE_SIZE d"]
      o = torch.zeros(q.shape, dtype=Q.dtype, device=Q.device)
      # type: Float[Tensor, "BATCH_SIZE Q_TILE_SIZE"]
      l = torch.zeros(q.shape[:-1], dtype=Q.dtype, device=Q.device)
      m = torch.full(q.shape[:-1], float("-inf"), dtype=Q.dtype, device=Q.device)
      for k_start in range(0, K.shape[1], K_TILE_SIZE):
        k_end = min(k_start + K_TILE_SIZE, K.shape[1])
        k = K[:, k_start:k_end]  # type: Float[Tensor, "BATCH_SIZE K_TILE_SIZE d"]
        v = V[:, k_start:k_end]  # type: Float[Tensor, "BATCH_SIZE K_TILE_SIZE d"]

        # compute tile of pre-softmax attention scores
        s = einsum(q, k, "batch bq d, batch bk d ->batch bq bk") * scale

        # compute logsumexp
        m_new = torch.maximum(m, torch.max(s, dim=-1).values)
        # type: Float[Tensor, "BATCH_SIZE Q_TILE_SIZE K_TILE_SIZE"]
        p = torch.exp(s - m_new.unsqueeze(-1))
        assert p.shape[-2:] == (Q_TILE_SIZE, K_TILE_SIZE)
        l_new = torch.exp(m - m_new) * l + torch.sum(p, dim=-1)

        o_new = einsum(o, torch.exp(m - m_new), "... bq d, ... bq -> ... bq d") + \
            einsum(p, v, "... bq bk, ... bk d -> ... bq d")

        # update O_ij, L_ij, M_ij
        o, l, m = o_new, l_new, m_new

      #
      O = einsum(o, l.reciprocal(), "... bq d, ... bq  -> ... bq d")
      L = m + torch.log(l)
      result_O.append(O)
      result_L.append(L)
    O = torch.cat(result_O, dim=1)
    L = torch.cat(result_L, dim=1)
    O = O.reshape(q_shape)
    L = L.reshape(q_shape[:-1])
    Q, K, V = (X.reshape(shape) for X, shape in zip((Q, K, V), (q_shape, k_shape, v_shape)))
    ctx.save_for_backward(Q, K, V, O, L)
    return O

  @staticmethod
  def backward(ctx, grad_output):
    pass


# %%
