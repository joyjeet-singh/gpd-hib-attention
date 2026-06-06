import jax
import jax.numpy as jnp
import time
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'code'))
from jax_mpo_attention import jax_mpo_attention_forward
import json

# ASSERT_CONVENTION: natural_units=natural, metric_signature=euclidean, fourier_convention=physics, coupling_convention=N/A, renormalization_scheme=N/A, gauge_choice=N/A

def profile_manual(n):
    d = 64
    chi = 8
    q = jax.random.normal(jax.random.PRNGKey(0), (n, d))
    k = jax.random.normal(jax.random.PRNGKey(1), (n, d))
    v = jax.random.normal(jax.random.PRNGKey(2), (n, d))
    
    # Measure launch time only
    start = time.time()
    for _ in range(100):
        jax_mpo_attention_forward(q, k, v, chi) # No block_until_ready
    launch_time = (time.time() - start) / 100 * 1000
    
    # Measure execution time
    start = time.time()
    for _ in range(100):
        jax_mpo_attention_forward(q, k, v, chi).block_until_ready()
    exec_time = (time.time() - start) / 100 * 1000
    
    compute_time = exec_time - launch_time
    
    return {"launch_time": launch_time, "compute_time": compute_time, "total_time": exec_time}

if __name__ == "__main__":
    results = {256: profile_manual(256), 2048: profile_manual(2048)}
    with open("analysis/bottleneck_analysis.json", "w") as f:
        json.dump(results, f, indent=4)
    print(results)
