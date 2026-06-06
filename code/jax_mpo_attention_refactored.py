# ASSERT_CONVENTION: natural_units=natural, metric_signature=euclidean, fourier_convention=physics, coupling_convention=N/A, renormalization_scheme=N/A, gauge_choice=N/A
import jax
import jax.numpy as jnp
from jax import lax

@jax.jit
def hierarchical_attention_block(q, k, v):
    n_blocks, block_size, d = q.shape
    attn = jnp.matmul(q, k.transpose(0, 2, 1)) / jnp.sqrt(d)
    attn = jax.nn.softmax(attn, axis=-1)
    out = jnp.matmul(attn, v)
    return jnp.mean(out, axis=1)

@jax.jit
def jax_mpo_attention_forward(q, k, v, L):
    n, d = q.shape
    
    # We contract tokens in groups of 4 iteratively.
    def fori_body(i, carry):
        q, k, v = carry
        
        q_b = q.reshape(-1, 4, d)
        k_b = k.reshape(-1, 4, d)
        v_b = v.reshape(-1, 4, d)
        
        out = hierarchical_attention_block(q_b, k_b, v_b)
        
        return (out, out, out)

    # Use lax.fori_loop which handles tracers for L!
    result = lax.fori_loop(0, L, fori_body, (q, k, v))
    
    return result[0]
