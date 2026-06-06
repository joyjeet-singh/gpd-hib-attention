# ASSERT_CONVENTION: natural_units=natural, metric_signature=euclidean, fourier_convention=physics, coupling_convention=N/A, renormalization_scheme=N/A, gauge_choice=N/A, coordinate_system=row-major
import torch
from transformers import GPT2Model, GPT2Config
import numpy as np
import os
import gc
import json

def extract_attention_weights(layer_idx, sequence_length):
    """
    Extract attention weights from GPT-2 Small for a given layer and sequence length.
    Memory-safe: deletes model after extraction.
    """
    print(f"Extracting attention weights for Layer {layer_idx}, n={sequence_length}...")
    
    # Load model
    model_name = "gpt2"
    model = GPT2Model.from_pretrained(model_name, output_attentions=True)
    model.eval()
    
    # Dummy input
    input_ids = torch.zeros((1, sequence_length), dtype=torch.long)
    
    with torch.no_grad():
        outputs = model(input_ids)
        # Take Head 0 of the specified layer.
        attention_matrix = outputs.attentions[layer_idx][0, 0].cpu().numpy()
        
    # Memory cleanup
    del model
    del outputs
    gc.collect()
    
    # Normalize: Tr[A] = 1
    trace = np.trace(attention_matrix)
    if trace > 0:
        attention_matrix /= trace
        
    return attention_matrix

def get_mpo_params(n, chi):
    L = int(np.log2(n))
    return 4 * L * chi * chi

def get_svd_rank_from_params(n, mpo_params):
    k = mpo_params / (2 * n + 1)
    return int(max(1, min(n, k)))

def compress_mps(A, bond_dim):
    n = A.shape[0]
    L = int(np.log2(n))
    
    tensor_shape = [2] * (2 * L)
    A_tensor = A.reshape(tensor_shape)
    
    permutation = []
    for k in range(L):
        permutation.append(k)      # i_k
        permutation.append(k + L)  # j_k
    A_mpo_format = A_tensor.transpose(permutation).reshape([4] * L)
    
    cores = []
    curr = A_mpo_format
    
    # MPS compression
    for i in range(L - 1):
        curr = curr.reshape(curr.shape[0], -1)
        u, s, vh = np.linalg.svd(curr, full_matrices=False)
        rank = min(bond_dim, len(s))
        u = u[:, :rank]
        s = s[:rank]
        vh = vh[:rank, :]
        cores.append(u.reshape(-1, rank))
        curr = (np.diag(s) @ vh).reshape(rank, -1)
    cores.append(curr)
    
    # Reconstruction
    recon = cores[0]
    for i in range(1, L):
        recon = np.tensordot(recon, cores[i], axes=(1, 0))
    
    # Corrected reconstruction shape handling
    A_mpo_recon = recon.reshape([2] * (2 * L))
    inv_permutation = [0] * (2 * L)
    # The permutation used in transpose was [i_0, j_0, i_1, j_1, ...]
    # Which corresponds to indices 0, L, 1, L+1, ...
    # The inv_permutation should be the reverse mapping to [i_0, i_1, ..., i_L, j_0, j_1, ..., j_L]
    # Re-doing the permutation logic:
    # A_mpo_format was constructed by transposing A_tensor to [0, L, 1, L+1, ..., L-1, 2L-1]
    # And then reshaping to [4, ..., 4].
    # So axis 0 of A_mpo_format corresponds to (i_0, j_0), axis 1 to (i_1, j_1), etc.
    # When reshaped to [2, 2, ..., 2, 2] (2L dimensions):
    # Axis 2k is i_k, Axis 2k+1 is j_k.
    # We want to reorder this to: i_0, i_1, ..., i_L-1, j_0, j_1, ..., j_L-1
    # i_0 is axis 0, i_1 is axis 2, ..., i_L-1 is axis 2*(L-1)
    # j_0 is axis 1, j_1 is axis 3, ..., j_L-1 is axis 2*(L-1)+1
    
    reorder_permutation = []
    for k in range(L):
        reorder_permutation.append(2*k)
    for k in range(L):
        reorder_permutation.append(2*k + 1)
        
    A_mpo = A_mpo_recon.transpose(reorder_permutation).reshape(n, n)
    
    return A_mpo

def mpo_compress_parameter_matched(A, chi):
    # MPO Compression
    A_mpo = compress_mps(A, chi)
    mpo_err = np.linalg.norm(A - A_mpo)
    
    # SVD Compression (Parameter Matched)
    n = A.shape[0]
    mpo_params = get_mpo_params(n, chi)
    svd_rank = get_svd_rank_from_params(n, mpo_params)
    
    u_svd, s_svd, vh_svd = np.linalg.svd(A, full_matrices=False)
    s_trunc = s_svd.copy()
    if svd_rank < len(s_trunc):
        s_trunc[svd_rank:] = 0
    A_svd = (u_svd * s_trunc) @ vh_svd
    svd_err = np.linalg.norm(A - A_svd)
    
    return mpo_err, svd_err, mpo_params, svd_rank

def main():
    layers = [0, 5, 11]
    n_values = [64, 128, 256]
    chi_values = [2, 4, 8, 16, 32]
    
    results = {}
    
    for n in n_values:
        results[str(n)] = {}
        for L in layers:
            npy_path = f"analysis/A_L{L}_n{n}.npy"
            if os.path.exists(npy_path):
                print(f"Loading attention weights for Layer {L}, n={n} from {npy_path}...")
                A = np.load(npy_path)
            else:
                A = extract_attention_weights(L, n)
                np.save(npy_path, A)
            
            results[str(n)][str(L)] = {}
            for chi in chi_values:
                mpo_err, svd_err, mpo_params, svd_rank = mpo_compress_parameter_matched(A, chi)
                delta_chi = mpo_err - svd_err
                results[str(n)][str(L)][str(chi)] = {
                    "mpo_err": float(mpo_err),
                    "svd_err": float(svd_err),
                    "delta_chi": float(delta_chi),
                    "mpo_params": int(mpo_params),
                    "svd_rank": int(svd_rank)
                }
                print(f"n={n}, L={L}, chi={chi} -> delta(chi)={delta_chi:.6f} (MPO={mpo_err:.6f}, SVD_rank={svd_rank}, SVD_err={svd_err:.6f})")
                
    with open("analysis/structural_gap_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Measurements complete. Results saved to analysis/structural_gap_results.json")

if __name__ == "__main__":
    os.makedirs("analysis", exist_ok=True)
    main()
