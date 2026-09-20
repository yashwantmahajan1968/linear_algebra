import numpy as np


def project_line(b, a):
    """Project vector b onto line through a.
       Returns: projection vector p = (aᵀb / aᵀa) * a"""
    dim = len(b)
    b = np.array(b,dtype='float').reshape(dim,1)
    a = np.array(a,dtype='float').reshape(dim,1)
    # print(a.shape,b.shape)
    return (a.T @ b * a) / (a.T@a)

def project_subspace(b, A):
    """Project b onto C(A).
       Returns: p = A @ inv(Aᵀ@A) @ Aᵀ @ b"""
    # Works for independent columns
    p = A @ np.linalg.inv(A.T@A) @ A.T @ b 
    return p

def projection_matrix(A):
    """Returns P = A @ inv(AᵀA) @ Aᵀ.
       Properties to verify: P² = P, Pᵀ = P."""
    p = A @ np.linalg.inv(A.T@A)@A.T
    return p

def least_squares(A, b):
    """Solve AᵀA x_hat = Aᵀb.
       Returns x_hat (the best approximate solution)."""

    return np.linalg.inv(A.T@A)@A.T@b

def fit_line(x, y):
    """Fit y = a + bx to data. Build the matrix [[1,x1],[1,x2],...].
       Return (a, b) via least_squares."""
    n = len(x)
    
    A = np.hstack((np.ones((n,1),dtype='float'), x))
    return least_squares(A,y)


def fit_poly(x, y, degree):
    """Fit polynomial of given degree. Build Vandermonde matrix.
       Return coefficients via least_squares."""
    A = np.hstack([x**i for i in range(degree + 1)])
    return least_squares(A,y)





## tests
for _ in range(100):
    A = np.random.randn(6, 3)
    b = np.random.randn(6)

    p = project_subspace(b, A)
    error = b - p
    P = projection_matrix(A)

    # projection is in column space
    # error is orthogonal to every column
    for j in range(A.shape[1]):
        assert abs(A[:, j] @ error) < 1e-10, "error not ⊥ column space"

    # P is idempotent
    assert np.allclose(P @ P, P), "P² ≠ P"

    # P is symmetric
    assert np.allclose(P, P.T), "P ≠ Pᵀ"


# least squares matches numpy
x = np.array([1, 2, 3, 4, 5], dtype=float)
y = 2*x + 1 + np.random.randn(5)*0.3
a, b_coeff = fit_line(x.reshape(-1,1), y.reshape(-1,1))
np_coeffs = np.polyfit(x, y, 1)
assert abs(b_coeff - np_coeffs[0]) < 0.01