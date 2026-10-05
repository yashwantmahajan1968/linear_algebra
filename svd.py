import numpy as np


def svd(A_orig,tol=1e-10):
    """Compute SVD: A = U @ Sigma @ Vt.
       Step 1: eigendecompose AᵀA → V and σ² (eigenvalues)
       Step 2: σ = sqrt of eigenvalues
       Step 3: U = A @ V @ diag(1/σ) for each nonzero σ
       Returns U, sigma (1D array), Vt."""
    A = A_orig.copy()
    # A@A.T = U@sigma@U.T -> shape = mxm , U has eigenvectors of A@A.T which is mxm , so U also mxm
    # A.T@A = V@sigma@V.T -> shape = nxn similarly V is nxn
    vals, V = np.linalg.eigh(A.T @ A)          # symmetric: real, orthonormal

    order = np.argsort(vals)[::-1]             # sort descending
    vals, V = vals[order], V[:, order]         # reorder values AND columns together

    r = int(np.sum(vals > tol * vals[0])) if vals[0] > 0 else 0   # rank: drop ~zero eigenvalues
    sigma = np.sqrt(vals[:r])                  # singular values
    V = V[:, :r]

    U = (A @ V) / sigma                        # u_i = A v_i / sigma_i (divides each column)
    return U, sigma, V.T


def low_rank_approx(A_orig, k):
    """Keep only top k singular values/vectors.
       A_k = U[:,:k] @ diag(sigma[:k]) @ Vt[:k,:]
       This is the best rank-k approximation (Eckart-Young)."""
    A = A_orig.copy()
    U,sigma,Vt = svd(A)
    m,n = A.shape[0],A.shape[1]
    lora = np.zeros((m,n),dtype=float)
    for i in range(k):
      lora += sigma[i] * (U[:,i:i+1]@Vt[i:i+1,:])
    return lora


def pseudoinverse(A_orig):
    """A⁺ = V @ diag(1/σ for nonzero σ) @ Uᵀ.
       Generalizes inverse to non-square / singular matrices."""
    U, sigma, Vt = svd(A)
    return (Vt.T / sigma) @ U.T



for _ in range(100):
    A = np.random.randn(5, 3)
    U, sigma, Vt = svd(A)

    # reconstruct
    Sigma = np.zeros((3, 3))
    for i in range(len(sigma)):
        Sigma[i, i] = sigma[i]
    assert np.allclose(U @ Sigma @ Vt, A), "UΣVᵀ ≠ A"

    # U and V have orthonormal columns
    # print(a)
    assert np.allclose(U[:, :len(sigma)].T @ U[:, :len(sigma)], np.eye(len(sigma)))
    assert np.allclose(Vt @ Vt.T, np.eye(3))

    # singular values match numpy
    sigma_np = np.linalg.svd(A, compute_uv=False)
    assert np.allclose(sorted(sigma, reverse=True), sorted(sigma_np, reverse=True))

    # Eckart-Young: rank-k approx is best
    for k in range(1, len(sigma)):
        A_k = low_rank_approx(A, k)
        err_k = np.linalg.norm(A - A_k, 'fro')
        # perturb randomly — error should increase
        noise = np.random.randn(*A_k.shape) * 0.01
        err_perturbed = np.linalg.norm(A - (A_k + noise), 'fro')
        # not always true due to tiny noise, but check sigma relationship
        assert abs(err_k - np.sqrt(sum(s**2 for s in sigma[k:]))) < 1e-8


print("all tests passed")