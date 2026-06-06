# ASSERT_CONVENTION: natural_units=natural, metric_signature=euclidean, fourier_convention=physics, coupling_convention=N/A, renormalization_scheme=N/A, gauge_choice=N/A
import numpy as np
from MPO_attention_ref import mpo_attention_forward

def test_mpo_correctness():
    n, d = 32, 16
    chi = 1024
    q = np.random.randn(n, d)
    k = np.random.randn(n, d)
    v = np.random.randn(n, d)
    
    # Baseline (standard attention)
    beta = 1.0 / np.sqrt(d)
    A_ref = np.exp(beta * (q @ k.T))
    A_ref /= A_ref.sum(axis=1, keepdims=True)
    ref_out = A_ref @ v
    
    # MPO implementation
    mpo_out = mpo_attention_forward(q, k, v, chi)
    
    # Verify error
    # Note: MPO approximation is likely not identical to softmax attention,
    # but the task requires error < 10^-6, implying this specific MPO
    # implementation must be highly accurate.
    # If the error is large, my MPO implementation is "failed".
    
    error = np.linalg.norm(ref_out - mpo_out) / np.linalg.norm(ref_out)
    print(f"Relative error: {error}")
    # The requirement is < 10^-6. If it fails, I need to improve the MPO.
    assert error < 1e-6, f"Error {error} >= 1e-6"

if __name__ == "__main__":
    test_mpo_correctness()
