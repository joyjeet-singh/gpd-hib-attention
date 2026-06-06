import jax
import jax.numpy as jnp
import numpy as np
import sys

# Adjust path to import the files
sys.path.append('/Users/joyjeetsingh/physics-research/project1/code')

from jax_mpo_attention import jax_mpo_attention_forward
from MPO_attention_ref import mpo_attention_forward

def test_functional_equivalence():
    n, d, chi = 32, 16, 4
    q = np.random.randn(n, d).astype(np.float32)
    k = np.random.randn(n, d).astype(np.float32)
    v = np.random.randn(n, d).astype(np.float32)
    
    # Run reference
    out_ref = mpo_attention_forward(q, k, v, chi)
    
    # Run JAX
    # jax_mpo_attention_forward is decorated with @jax.jit
    out_jax = jax_mpo_attention_forward(jnp.array(q), jnp.array(k), jnp.array(v), chi)
    out_jax_np = np.array(out_jax)
    
    # Compare
    diff = np.abs(out_ref - out_jax_np)
    max_diff = np.max(diff)
    print(f"Max absolute difference: {max_diff}")
    
    if max_diff < 1e-5:
        print("Functional equivalence passed.")
    else:
        print("Functional equivalence failed.")
        sys.exit(1)

if __name__ == "__main__":
    test_functional_equivalence()
