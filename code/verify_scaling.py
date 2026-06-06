# ASSERT_CONVENTION: natural_units=natural, metric_signature=euclidean, fourier_convention=physics, coupling_convention=N/A, renormalization_scheme=N/A, gauge_choice=N/A
import numpy as np

def verify_scaling(alpha, chi, num_samples=100000):
    # Sum tail for sigma_i ~ i^-alpha
    # eps^2 = sum_{i=chi+1}^inf sigma_i^2 = sum_{i=chi+1}^inf i^(-2alpha)
    
    # Calculate sum approximately
    sum_tail = np.sum([i**(-2*alpha) for i in range(chi + 1, num_samples)])
    
    # Integral approximation: chi^(1-2alpha) / (2alpha - 1)
    integral_val = (chi**(1 - 2*alpha)) / (2*alpha - 1)
    
    ratio = sum_tail / integral_val
    return ratio

alpha = 0.8
chi = 1000
ratio = verify_scaling(alpha, chi)
print(f'Ratio for alpha={alpha}, chi={chi}: {ratio}')

alpha = 0.9
chi = 500
ratio = verify_scaling(alpha, chi)
print(f'Ratio for alpha={alpha}, chi={chi}: {ratio}')
