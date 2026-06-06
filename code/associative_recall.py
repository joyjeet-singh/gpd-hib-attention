# ASSERT_CONVENTION: natural_units=natural, metric_signature=euclidean, fourier_convention=physics, coupling_convention=N/A, renormalization_scheme=N/A, gauge_choice=N/A
import numpy as np
import json
from MPO_attention_ref import mpo_attention_forward

def run_associative_recall_trial(n, d, chi):
    # n/2 pairs
    num_pairs = n // 2
    keys = np.random.randn(num_pairs, d)
    values = np.random.randn(num_pairs, d)
    
    # Randomly select a query key from the set
    idx = np.random.randint(0, num_pairs)
    query_key = keys[idx].reshape(1, d)
    target_value = values[idx]
    
    # Run MPO attention
    # Construct sequence of all pairs for the attention mechanism
    # Padding to n if necessary, but assuming mpo_attention_forward handles n length keys
    # Actually, for standard attention, keys would be n x d. 
    # Let's construct a full sequence of length n.
    full_keys = np.random.randn(n, d)
    full_values = np.random.randn(n, d)
    # Insert pairs
    full_keys[:num_pairs] = keys
    full_values[:num_pairs] = values
    
    output = mpo_attention_forward(query_key, full_keys, full_values, chi)
    
    # Check if output is closest to target_value
    # Assuming standard attention over full_values.
    # The output of attention is a weighted sum of values.
    # For a sharp softmax, it should retrieve the target_value.
    
    # Measure: is the output closer to target_value than to any other value?
    # Or simply: is the norm of difference small?
    error = np.linalg.norm(output.flatten() - target_value.flatten())
    
    # Threshold for correctness
    return error < 0.1

def run_evaluation():
    d = 64
    trials = 100
    results = {}
    
    for n in [64, 128, 256, 512]:
        results[f"n={n}"] = {}
        for chi in [2, 4, 8, 16]:
            correct = 0
            for _ in range(trials):
                if run_associative_recall_trial(n, d, chi):
                    correct += 1
            accuracy = correct / trials
            results[f"n={n}"][f"chi={chi}"] = accuracy
            print(f"n={n}, chi={chi}: {accuracy}")
            
    with open("analysis/synthetic_task_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_evaluation()
