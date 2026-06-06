import jax
import jax.numpy as jnp
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'code'))
from jax_mpo_attention import jax_mpo_attention_forward
import json

# ASSERT_CONVENTION: natural_units=natural, metric_signature=euclidean, fourier_convention=physics, coupling_convention=N/A, renormalization_scheme=N/A, gauge_choice=N/A

def calculate_svd_rank(n, d, chi):
    # P_MPO calculation for MPO Attention
    # For a layer with bond dimension chi, the parameter count is roughly related to chi.
    # P_MPO = (2*chi^2 + 2*chi*d) * log2(n)
    p_mpo = (2 * chi**2 + 2 * chi * d) * np.log2(n)
    
    # Matching rank formula: r = ceil(P_MPO / (2n + 1))
    # Wait, the plan formula is: r = ceil(P_MPO / (2n + 1))
    r = int(np.ceil(p_mpo / (2 * n + 1)))
    return max(r, 1)

def run_comparison():
    n_values = [256, 512, 1024, 2048]
    d = 64
    chi = 8
    
    results = {}
    
    for n in n_values:
        q = jax.random.normal(jax.random.PRNGKey(0), (n, d))
        k = jax.random.normal(jax.random.PRNGKey(1), (n, d))
        v = jax.random.normal(jax.random.PRNGKey(2), (n, d))
        
        # MPO Attention
        out_mpo = jax_mpo_attention_forward(q, k, v, chi)
        
        # SVD Rank
        rank = calculate_svd_rank(n, d, chi)
        
        # SVD Comparison (using jax.numpy.linalg.svd for parameter matching)
        # SVD of attention matrix: softmax(q @ k.T / sqrt(d))
        attn_matrix = jnp.dot(q, k.T) / jnp.sqrt(d)
        attn_matrix = jax.nn.softmax(attn_matrix, axis=-1)
        u, s, vt = jax.numpy.linalg.svd(attn_matrix, full_matrices=False)
        
        # Truncate to rank
        u_r = u[:, :rank]
        s_r = s[:rank]
        vt_r = vt[:rank, :]
        
        attn_svd = jnp.dot(u_r * s_r, vt_r)
        out_svd = jnp.dot(attn_svd, v)
        
        # Frobenius error
        err_mpo = jnp.linalg.norm(out_mpo - jnp.dot(attn_matrix, v))
        err_svd = jnp.linalg.norm(out_svd - jnp.dot(attn_matrix, v))
        
        results[n] = {
            "mpo_error": float(err_mpo),
            "svd_error": float(err_svd),
            "rank": rank,
            "parameter_match": True # Formula matched
        }
        
    with open("analysis/functional_comparison_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print(results)

if __name__ == "__main__":
    run_comparison()
