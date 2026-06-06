# ASSERT_CONVENTION: natural_units=natural, metric_signature=euclidean, fourier_convention=physics, coupling_convention=N/A, renormalization_scheme=N/A, gauge_choice=N/A
import math

def calculate_chi_schedule(n, k, alpha, max_l):
    """
    Calculates the bond dimension chi(l) for each RG scale l.
    
    Args:
        n (int): Initial sequence length (UV scale l=0).
        k (int): Coarse-graining factor.
        alpha (float): Scaling exponent (c/3 for critical systems).
        max_l (int): Maximum RG scale.
        
    Returns:
        list: Bond dimensions at each scale l.
    """
    schedule = []
    for l in range(max_l + 1):
        n_l = n / (k**l)
        # chi(l) ~ n_l^alpha
        # We ensure chi is at least 1
        chi_l = max(1, int(math.ceil(n_l**alpha)))
        schedule.append(chi_l)
    return schedule

def error_bounded_chi(epsilon, gamma):
    """
    Determines required chi for a fixed error epsilon given singular value decay gamma.
    sigma_i ~ i^-gamma
    epsilon^2 ~ chi^(1-2*gamma)
    """
    if gamma <= 0.5:
        raise ValueError("Gamma must be > 0.5 for convergent error sum.")
    
    # chi ~ epsilon^(2/(1-2*gamma))
    exponent = 2.0 / (1.0 - 2.0 * gamma)
    chi = epsilon**exponent
    return int(math.ceil(chi))

if __name__ == "__main__":
    n = 1024
    k = 2
    alpha = 0.5  # Example: moderately entangled
    max_l = int(math.log2(n))
    
    print(f"RG Scale Bond Dimension Schedule (n={n}, k={k}, alpha={alpha}):")
    schedule = calculate_chi_schedule(n, k, alpha, max_l)
    for l, chi in enumerate(schedule):
        n_l = n / (k**l)
        print(f"Scale l={l:2d} | Effective n_l={n_l:7.1f} | Bond Dimension chi={chi:4d}")

    # Asymptotic check
    print("\nAsymptotic behavior check (large n, l=0):")
    for large_n in [1024, 4096, 16384]:
        chi_0 = calculate_chi_schedule(large_n, k, alpha, 0)[0]
        print(f"n={large_n:6d} | chi(0)={chi_0:4d} | ratio chi/n={chi_0/large_n:.4f}")
