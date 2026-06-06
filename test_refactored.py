import jax
import jax.numpy as jnp
import time
import sys
sys.path.append("/Users/joyjeetsingh/physics-research/project1/code")
from jax_mpo_attention_refactored import jax_mpo_attention_forward

n = 1024
d = 64
L = int(jnp.log2(n)/2) # Based on plan: L = log2(n)/2
key = jax.random.PRNGKey(0)
q = jax.random.normal(key, (n, d))
k = jax.random.normal(key, (n, d))
v = jax.random.normal(key, (n, d))

# Warmup
jax_mpo_attention_forward(q, k, v, L).block_until_ready()

# Measure
start = time.time()
for _ in range(10):
    jax_mpo_attention_forward(q, k, v, L).block_until_ready()
end = time.time()
print(f"Time for n={n}: {(end - start)/10 * 1000:.2f} ms")
