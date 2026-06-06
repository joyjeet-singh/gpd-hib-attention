# ASSERT_CONVENTION: natural_units=natural, metric_signature=euclidean, fourier_convention=physics, coupling_convention=N/A, renormalization_scheme=N/A, gauge_choice=N/A
import jax
import jax.numpy as jnp
from functools import partial

@jax.jit
def mpo_block_contraction(q, k, v):
    # q: [n_blocks, 4, d]
    # k: [n_blocks, 4, d]
    # v: [n_blocks, 4, d]
    # This block performs the contraction A_ij = prod_l exp(q_il * k_jl * beta)
    # But doing it hierarchically.
    
    n_blocks, block_size, d = q.shape
    beta = 1.0 / jnp.sqrt(d)
    
    # Hierarchical product of exp(q_il * k_jl * beta)
    # A_ij = exp(sum_l q_il * k_jl * beta)
    
    # Contraction over dimensions (l)
    attn = jnp.exp(jnp.einsum('bid,bjd->bij', q, k) * beta)
    # Normalization (as per MPO_attention_ref)
    attn /= attn.sum(axis=-1, keepdims=True)
    
    # Contraction with V
    out = jnp.einsum('bij,bjd->bid', attn, v)
    
    # Return [n_blocks, d] (mean over block size 4)
    return jnp.mean(out, axis=1)

@partial(jax.jit, static_argnums=(3,))
def jax_mpo_attention_unrolled(q, k, v, L):
    """
    Refactored kernel using static Python loop unrolling for tree depth L.
    """
    L = int(L)
    curr_q, curr_k, curr_v = q, k, v
    
    for i in range(L):
        curr_q_b = curr_q.reshape(-1, 4, curr_q.shape[-1])
        curr_k_b = curr_k.reshape(-1, 4, curr_k.shape[-1])
        curr_v_b = curr_v.reshape(-1, 4, curr_v.shape[-1])
        
        out = mpo_block_contraction(curr_q_b, curr_k_b, curr_v_b)
        
        curr_q = out
        curr_k = out
        curr_v = out
        
    return curr_q
