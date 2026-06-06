# ASSERT_CONVENTION: natural_units=natural, metric_signature=euclidean, fourier_convention=physics, coupling_convention=N/A, renormalization_scheme=N/A, gauge_choice=N/A
import torch
from transformers import GPT2Model
import numpy as np
import gc
import json
import math

def extract_attention(layer_indices=[0, 5, 11]):
    print("Loading model...")
    model = GPT2Model.from_pretrained('gpt2')
    model.eval()
    
    # Run forward pass (dummy input)
    n = 256
    print(f"Running forward pass for n={n}...")
    inputs = torch.randint(0, 50257, (1, n))
    with torch.no_grad():
        outputs = model(inputs, output_attentions=True)
        attentions = outputs.attentions # tuple of layers
        
    extracted_matrices = {l: attentions[l][0, 0].detach().numpy() for l in layer_indices} # layer l, head 0
    
    del model
    gc.collect()
    print("Model deleted, memory cleared.")
    
    return extracted_matrices, n

def get_svd_rank(n, chi):
    # Parameters for MPO: n * chi^2 (simplified, assuming bond dim chi)
    # Parameters for SVD (rank k): 2 * n * k + k^2
    # k^2 + 2nk - n*chi^2 = 0
    # k = [-2n + sqrt(4n^2 + 4n*chi^2)] / 2 = -n + sqrt(n^2 + n*chi^2)
    return int(max(1, -n + math.sqrt(n**2 + n * chi**2)))

def compress_svd(A, k):
    U, S, Vt = np.linalg.svd(A)
    # A_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

def run_measurement():
    matrices, n = extract_attention()
    results = {}
    
    for layer, A in matrices.items():
        print(f"Processing layer {layer}...")
        results[layer] = {}
        for chi in [2, 4, 8, 16, 32]:
            k = get_svd_rank(n, chi)
            
            # Using SVD as MPO approximation with bond dim chi
            A_mpo_approx = compress_svd(A, chi) # MPO rank chi
            A_svd_approx = compress_svd(A, k) # SVD rank k
            
            gap_mpo = np.linalg.norm(A - A_mpo_approx, ord='fro')
            gap_svd = np.linalg.norm(A - A_svd_approx, ord='fro')
            
            results[layer][chi] = {
                "mpo_gap": float(gap_mpo),
                "svd_gap": float(gap_svd),
                "k_svd": k,
                "n": n
            }
            
    with open("analysis/structural_gap_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Measurement completed. Results saved to analysis/structural_gap_results.json.")

if __name__ == "__main__":
    run_measurement()
