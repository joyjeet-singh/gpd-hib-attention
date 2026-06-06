# ASSERT_CONVENTION: natural_units=natural, metric_signature=euclidean, fourier_convention=physics, coupling_convention=N/A, renormalization_scheme=N/A, gauge_choice=N/A
import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

class MERA_Attention_NumPy:
    def __init__(self, d_model, chi_0=16, alpha=0.5):
        self.d_model = d_model
        self.chi_0 = chi_0
        self.alpha = alpha
        
        self.chi_1 = max(1, int(chi_0))
        self.chi_2 = max(1, int(chi_0 * (0.5**alpha)))
        
        # Initialize Isometries and Projections
        # Enforcing W W^T = I
        self.W1, _ = np.linalg.qr(np.random.normal(size=(2 * d_model, self.chi_1)))
        self.W1 = self.W1.T # [chi_1, 2*d_model]
        
        self.W2, _ = np.linalg.qr(np.random.normal(size=(2 * self.chi_1, self.chi_2)))
        self.W2 = self.W2.T # [chi_2, 2*chi_1]
        
        self.wq = np.random.normal(size=(d_model, d_model)) / np.sqrt(d_model)
        self.wk = np.random.normal(size=(d_model, d_model)) / np.sqrt(d_model)
        self.wv = np.random.normal(size=(d_model, d_model)) / np.sqrt(d_model)

    def forward(self, x):
        n, d = x.shape
        # Initial Projections
        Q0 = x @ self.wq
        K0 = x @ self.wk
        V0 = x @ self.wv
        
        # Level 0
        q0 = Q0.reshape(n // 2, 2, d)
        k0 = K0.reshape(n // 2, 2, d)
        v0 = V0.reshape(n // 2, 2, d)
        attn0 = np.matmul(q0, k0.transpose(0, 2, 1)) / np.sqrt(d)
        attn0 = softmax(attn0)
        out0 = np.matmul(attn0, v0).reshape(n, d)
        
        # Level 1
        Q1 = Q0.reshape(n // 2, 2 * d) @ self.W1.T
        K1 = K0.reshape(n // 2, 2 * d) @ self.W1.T
        V1 = V0.reshape(n // 2, 2 * d) @ self.W1.T
        
        q1 = Q1.reshape(n // 4, 2, self.chi_1)
        k1 = K1.reshape(n // 4, 2, self.chi_1)
        v1 = V1.reshape(n // 4, 2, self.chi_1)
        
        attn1 = np.matmul(q1, k1.transpose(0, 2, 1)) / np.sqrt(self.chi_1)
        attn1 = softmax(attn1)
        out1_c = np.matmul(attn1, v1).reshape(n // 2, self.chi_1)
        out1 = (out1_c @ self.W1).reshape(n, d)
        
        # Level 2
        Q2 = Q1.reshape(n // 4, 2 * self.chi_1) @ self.W2.T
        K2 = K1.reshape(n // 4, 2 * self.chi_1) @ self.W2.T
        V2 = V1.reshape(n // 4, 2 * self.chi_1) @ self.W2.T
        
        attn2 = np.matmul(Q2, K2.T) / np.sqrt(self.chi_2)
        attn2 = softmax(attn2)
        out2_c2 = np.matmul(attn2, V2)
        
        out2_c1 = (out2_c2 @ self.W2).reshape(n // 2, self.chi_1)
        out2 = (out2_c1 @ self.W1).reshape(n, d)
        
        return out0 + out1 + out2
