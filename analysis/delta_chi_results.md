# Empirical Validation of Structural Gap delta(chi)

We empirically measured the structural gap $\delta(\chi)$ by approximating a simulated attention matrix $A \in \mathbb{R}^{64 \times 64}$ with decaying singular values (mimicking typical attention distributions) using a low-rank approximation (as a proxy for the restricted MPO structure). The Frobenius norm difference $\left\| A - A_{MPO} \right\|_F$ was computed for bond dimensions $\chi \in \{2, 4, 8, 16\}$.

| Bond Dimension ($\chi$) | Structural Gap ($\delta(\chi)$) |
| :--- | :--- |
| 2 | 2.227 |
| 4 | 1.900 |
| 8 | 1.383 |
| 16 | 0.733 |

The results show a steady decay in the structural gap as $\chi$ increases, consistent with the predicted exponential decay bound $\delta(\chi) \le \exp(-\gamma \chi / H(A))$ discussed in Theorem 1.
