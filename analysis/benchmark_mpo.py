import jax
import jax.numpy as jnp
import time
import numpy as np
import sys
import os

# Add code directory to path
sys.path.append(os.path.join(os.getcwd(), 'code'))
from jax_mpo_attention import jax_mpo_attention_forward

def benchmark():
    n_values = [64, 128, 256, 512, 1024, 2048]
    d = 64
    chi = 32
    
    results = []
    for n in n_values:
        q = jax.random.normal(jax.random.PRNGKey(0), (n, d))
        k = jax.random.normal(jax.random.PRNGKey(1), (n, d))
        v = jax.random.normal(jax.random.PRNGKey(2), (n, d))
        
        # Warmup (more iterations)
        for _ in range(50):
            jax_mpo_attention_forward(q, k, v, chi).block_until_ready()
        
        # Time
        start = time.time()
        for _ in range(100):
            jax_mpo_attention_forward(q, k, v, chi).block_until_ready()
        end = time.time()
        results.append((n, (end - start) / 100 * 1000))
    return results

if __name__ == "__main__":
    print(benchmark())
