# Project Conventions

## General Conventions

- **Natural Units:** `natural`
  - *Rationale:* Simplifies many equations by setting c=hbar=kB=1.
- **Metric Signature:** `euclidean`
  - *Rationale:* Standard for Euclidean tensor network representations.
- **Fourier Convention:** `physics` (e.g., ∫ dp / (2π)^d e^(ipx))
  - *Rationale:* Consistent with common QFT textbooks.
- **Index Positioning:** `Einstein` (summation convention for repeated indices)
  - *Rationale:* Standard for tensor notation.
- **State Normalization:** `trace-normalized` (Tr[ρ] = 1)
  - *Rationale:* Appropriate for density matrix representations in tensor networks.
- **Levi-Civita Sign:** `+1` (for ε^0123)
  - *Rationale:* Standard choice in 4D spacetime.
- **Generator Normalization:** `delta/2` (for SU(N) generators [T_a, T_b] = i f_abc T_c, Tr(T_a T_b) = delta_ab / 2)
  - *Rationale:* Standard for non-abelian gauge theories.
- **Creation/Annihilation Operator Ordering:** `normal` (annihilation operators to the right of creation operators in Wick's theorem)
  - *Rationale:* Standard for vacuum expectation values and correlators.

## Unset Conventions

The following convention fields are currently unset and may need to be defined as the project progresses:
- Gauge Choice
- Regularization Scheme
- Renormalization Scheme
- Coordinate System
- Spin Basis
- Coupling Convention
- Time Ordering
- Commutation Convention
- Covariant Derivative Sign
- Gamma Matrix Convention

These will be addressed as needed during the research phases.
