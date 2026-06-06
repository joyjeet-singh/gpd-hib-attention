import jax
import jax.numpy as jnp
import time
import sys
import os
sys.path.append(os.path.abspath("/Users/joyjeetsingh/physics-research/project1/code"))

from jax_mpo_attention import jax_mpo_attention_forward

# Standard Attention Baseline
@jax.jit
def jax_standard_attention(q, k, v):
    d = q.shape[-1]
    # Standard attention: softmax(QK^T / sqrt(d)) * V
    A = jax.nn.softmax((q @ k.T) / jnp.sqrt(d), axis=-1)
    return A @ v

def benchmark():
    n_values = [64, 128, 256, 512, 1024, 2048]
    d = 64
    
    results = []
    print("n | JAX MPO (ms) | JAX Std (ms) | Speedup")
    for n in n_values:
        key = jax.random.PRNGKey(0)
        q = jax.random.normal(key, (n, d))
        k = jax.random.normal(key, (n, d))
        v = jax.random.normal(key, (n, d))
        
        # Warmup
        jax_mpo_attention_forward(q, k, v, 4).block_until_ready()
        jax_standard_attention(q, k, v).block_until_ready()
        
        # Measure MPO
        start = time.time()
        for _ in range(10):
            jax_mpo_attention_forward(q, k, v, 4).block_until_ready()
        mpo_time = (time.time() - start) / 10 * 1000
        
        # Measure Standard
        start = time.time()
        for _ in range(10):
            jax_standard_attention(q, k, v).block_until_ready()
        std_time = (time.time() - start) / 10 * 1000
        
        print(f"{n} | {mpo_time:.2f} | {std_time:.2f} | {std_time/mpo_time:.2f}x")
        results.append((n, mpo_time, std_time, std_time/mpo_time))
            
    return results

if __name__ == "__main__":
    benchmark()
