import numpy as np

def mpo_compress_check(A, chi):
    n = A.shape[0]
    L = int(np.log2(n))
    tensor_shape = [2] * (2 * L)
    A_tensor = A.reshape(tensor_shape)
    permutation = []
    for k in range(L):
        permutation.append(k)      # i_k
        permutation.append(k + L)  # j_k
    A_mpo_format = A_tensor.transpose(permutation).reshape([4] * L)
    
    def compress_mps(vec, bond_dim):
        cores = []
        curr = vec
        for i in range(L - 1):
            curr = curr.reshape(-1, 4**(L-1-i))
            u, s, vh = np.linalg.svd(curr, full_matrices=False)
            rank = min(bond_dim, len(s))
            u = u[:, :rank]
            s = s[:rank]
            vh = vh[:rank, :]
            cores.append(u)
            curr = np.diag(s) @ vh
        cores.append(curr)
        recon = cores[0]
        for i in range(1, L):
            recon = recon @ cores[i]
        return recon.reshape([4] * L)

    A_mpo_recon_flat = compress_mps(A_mpo_format, chi)
    A_mpo_recon_tensor = A_mpo_recon_flat.reshape([2] * (2 * L))
    inv_permutation = [0] * (2 * L)
    for k in range(L):
        inv_permutation[k] = 2 * k
        inv_permutation[k + L] = 2 * k + 1
    A_mpo = A_mpo_recon_tensor.transpose(inv_permutation).reshape(n, n)
    return np.linalg.norm(A - A_mpo)

def svd_compress_check(A, rank):
    u, s, vh = np.linalg.svd(A, full_matrices=False)
    s_trunc = s.copy()
    if rank < len(s_trunc):
        s_trunc[rank:] = 0
    A_svd = (u * s_trunc) @ vh
    return np.linalg.norm(A - A_svd)

# Test with a random matrix
np.random.seed(42)
n = 64
A = np.random.randn(n, n)
A /= np.trace(A) # Normalize like in the report

for chi in [2, 4, 8]:
    mpo_err = mpo_compress_check(A, chi)
    svd_err = svd_compress_check(A, chi)
    print(f"chi={chi}: MPO_err={mpo_err:.6f}, SVD_err={svd_err:.6f}, delta={mpo_err-svd_err:.6f}")

# Test with a low-rank matrix + noise
A_low = np.outer(np.random.randn(n), np.random.randn(n))
A_noise = A_low + 0.1 * np.random.randn(n, n)
A_noise /= np.trace(A_noise)

print("\nLow-rank + noise:")
for chi in [1, 2]:
    mpo_err = mpo_compress_check(A_noise, chi)
    svd_err = svd_compress_check(A_noise, chi)
    print(f"chi={chi}: MPO_err={mpo_err:.6f}, SVD_err={svd_err:.6f}, delta={mpo_err-svd_err:.6f}")

