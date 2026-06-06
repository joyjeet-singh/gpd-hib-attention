# ASSERT_CONVENTION: natural_units=natural, metric_signature=euclidean, fourier_convention=physics, coupling_convention=N/A, renormalization_scheme=N/A, gauge_choice=N/A
import numpy as np
from MPO_attention_ref import mpo_attention_forward

def multi_hop_reasoning(n, d, chi, k):
    # Synthetic multi-hop: node0 -> node1 -> node2 -> ... -> nodek
    
    # Generate nodes
    nodes = np.random.randn(k+1, d)
    keys = nodes[:-1]
    values = nodes[1:]
    
    # Initial query
    query = nodes[0].reshape(1, d)
    
    # Increase sharpness
    scaling = 5.0
    
    # Iterative reasoning: perform attention to "hop" from node i to node i+1
    for _ in range(k):
        query = mpo_attention_forward(query * np.sqrt(scaling), keys * np.sqrt(scaling), values, chi)
        
    # Check if final query is close to nodes[k]
    error = np.linalg.norm(query - nodes[k])
    return error

if __name__ == "__main__":
    # Constraints: n=256, d=64
    for k in [2, 3, 4]:
        print(f"Running multi-hop reasoning k={k}")
        error = multi_hop_reasoning(256, 64, 1024, k)
        print(f"Multi-hop reasoning k={k} error: {error:.6f}")
        assert error < 0.5, f"Reasoning error too high for k={k}: {error}"
        print(f"Multi-hop k={k} passed.")
