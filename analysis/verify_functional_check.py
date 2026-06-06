import json

def verify_structural_gap():
    with open('/Users/joyjeetsingh/physics-research/project1/analysis/structural_gap_results.json', 'r') as f:
        data = json.load(f)
    
    all_passed = True
    for n, layers in data.items():
        for l, chis in layers.items():
            for chi, results in chis.items():
                if results['delta_chi'] > 0:
                    print(f"FAILED: n={n}, L={l}, chi={chi}, delta_chi={results['delta_chi']}")
                    all_passed = False
                
    if all_passed:
        print("PASS: All empirical structural gaps are non-positive (delta_chi <= 0).")
    else:
        print("FAIL: Some structural gaps are positive.")

verify_structural_gap()
