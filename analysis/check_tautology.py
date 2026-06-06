import numpy as np
from scipy.linalg import svd

# 1. Define a dummy matrix A (e.g., 4x4)
A = np.random.rand(4, 4)
A = A / np.trace(A) # Trace-normalized

# 2. Simulate MPO compression (for rank chi=1)
# Simply take a rank-1 approximation for MPO (simplified)
U, S, Vh = svd(A)
A_MPO = S[0] * np.outer(U[:, 0], Vh[0, :])

# 3. Optimal rank-1 SVD
A_SVD = S[0] * np.outer(U[:, 0], Vh[0, :]) # In this simple rank-1 case, they are identical

# 4. Definition of delta
delta = np.linalg.norm(A - A_MPO) - np.linalg.norm(A - A_SVD)

print(f"Norm(A - A_MPO) = {np.linalg.norm(A - A_MPO)}")
print(f"Norm(A - A_SVD) = {np.linalg.norm(A - A_SVD)}")
print(f"delta = {delta}")
