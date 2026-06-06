import jax
import jax.numpy as jnp
import time
import numpy as np
import sys
import os
import json

# Add code directory to path
sys.path.append(os.path.join(os.getcwd(), 'code'))
from jax_mpo_attention import jax_mpo_attention_forward

# ASSERT_CONVENTION: natural_units=natural, metric_signature=euclidean, fourier_convention=physics, coupling_convention=N/A, renormalization_scheme=N/A, gauge_choice=N/A

def benchmark():
    n_values = [256, 512, 1024, 2048]
    d = 64
    chi = 8 # Matching parameter count as in functional task
    
    results = {}
    
    for n in n_values:
        q = jax.random.normal(jax.random.PRNGKey(0), (n, d))
        k = jax.random.normal(jax.random.PRNGKey(1), (n, d))
        v = jax.random.normal(jax.random.PRNGKey(2), (n, d))
        
        # MPO Attention
        start = time.time()
        for _ in range(10):
            jax_mpo_attention_forward(q, k, v, chi).block_until_ready()
        mpo_time = (time.time() - start) / 10 * 1000
        
        # Naive Attention
        def naive_attention(q, k, v):
            attn = jnp.dot(q, k.T) / jnp.sqrt(d)
            attn = jax.nn.softmax(attn, axis=-1)
            return jnp.dot(attn, v)
            
        start = time.time()
        for _ in range(10):
            naive_attention(q, k, v).block_until_ready()
        naive_time = (time.time() - start) / 10 * 1000
        
        # FA2 Proxy (jax.nn.dot_product_attention)
        start = time.time()
        for _ in range(10):
            jax.nn.dot_product_attention(q, k, v).block_until_ready()
        fa2_time = (time.time() - start) / 10 * 1000
        
        results[n] = {
            "mpo_time": mpo_time,
            "naive_time": naive_time,
            "fa2_time": fa2_time
        }
        
    with open("analysis/benchmark_mpo_fa2_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print(results)

if __name__ == "__main__":
    benchmark()
