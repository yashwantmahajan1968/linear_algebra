
import numpy as np

def gram_schmidt(V_orig):
    """Takes matrix V, returns Q with orthonormal columns.
       Process column by column:
         q_i = v_i - proj(v_i onto q_1) - proj(v_i onto q_2) - ...
         q_i = q_i / ||q_i||"""
    V = V_orig.copy()
    n = V.shape[1]
    
    for i in range(1,n):
      for j in range(i):
        V[:,i] = V[:,i] - ((V[:,j].T @ V[:,i])/(V[:,j].T @ V[:,j])) * V[:,j]
    for i in range(n):
      V[:,i] = V[:,i]/np.sqrt(V[:,i].T@V[:,i])
    return V

## can be completed later , not worth much time now
def qr(A):
    """Returns Q, R where A = QR.
       Q from gram_schmidt. R = QᵀA (upper triangular)."""

def qr_solve(A, b):
    """Solve Ax = b via QR: Rx = Qᵀb (triangular solve)."""


## tests
for _ in range(100):
    A = np.random.randn(5, 3)
    Q = gram_schmidt(A)

    assert np.allclose(Q.T @ Q, np.eye(3)), "QᵀQ ≠ I"
# for _ in range(100):
#     A = np.random.randn(5, 3)
#     Q, R = qr(A)

#     assert np.allclose(Q.T @ Q, np.eye(3)), "QᵀQ ≠ I"
#     assert np.allclose(Q @ R, A), "QR ≠ A"
#     # R is upper triangular
#     assert np.allclose(R, np.triu(R)), "R not upper triangular"

# # Stability test: ill-conditioned system
# A_bad = np.array([[1, 1], [1, 1.0001], [1, 1.0002]])
# b = np.array([2, 2.0001, 2.0003])
# x_normal = np.linalg.solve(A_bad.T @ A_bad, A_bad.T @ b)  # may be inaccurate
# x_qr = qr_solve(A_bad, b)  # should be better