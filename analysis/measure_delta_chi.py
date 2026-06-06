import numpy as np
from scipy.linalg import svd

# Simulate a standard attention matrix for n=64
n = 64
d = 64
# Create a matrix with decaying singular values to mimic attention structure
U, _, V = svd(np.random.randn(n, n))
s = np.exp(-np.linspace(0, 5, n))
A = U @ np.diag(s) @ V

def mpo_approx(A, chi):
    # Simplified low-rank approximation (SVD based)
    U, S, Vt = svd(A, full_matrices=False)
    A_approx = U[:, :chi] @ np.diag(S[:chi]) @ Vt[:chi, :]
    return A_approx

chi_values = [2, 4, 8, 16]
results = {}

for chi in chi_values:
    A_mpo = mpo_approx(A, chi)
    diff = np.linalg.norm(A - A_mpo, ord='fro')
    results[chi] = diff

print(results)
