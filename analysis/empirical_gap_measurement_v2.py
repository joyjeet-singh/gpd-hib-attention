import jax
import jax.numpy as jnp
import numpy as np
import time
from code.jax_mpo_attention_unrolled import jax_mpo_attention_unrolled
import json

def benchmark():
    results = {}
    n_list = [64, 128, 256, 512, 1024]
    
    for n in n_list:
        d = 8
        L = int(np.log2(n) / 2)
        q = jnp.array(np.random.randn(n, d).astype(np.float32))
        k = jnp.array(np.random.randn(n, d).astype(np.float32))
        v = jnp.array(np.random.randn(n, d).astype(np.float32))
        
        # Warm-up
        jax_mpo_attention_unrolled(q, k, v, L)
        
        # Benchmark
        start = time.time()
        for _ in range(10):
            res = jax_mpo_attention_unrolled(q, k, v, L)
            res.block_until_ready()
        end = time.time()
        
        results[n] = (end - start) / 10
        print(f"n={n}, time={results[n]}")
    
    with open("analysis/benchmarking_results_v2.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    benchmark()
