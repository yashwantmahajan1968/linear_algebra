import numpy as np, sympy
from scipy.linalg import null_space


TOL = 1e-10



def rref(A_orig):
    """Reduced row echelon form. Most important function you'll write.
       Returns: R (the RREF), E (the transformation matrix), pivot_cols (list of pivot column indices)."""

    A = A_orig.copy()
    m, n = len(A),len(A[0])
    E = np.eye(m,dtype='float')
    j = 0
    pivot_cols = []
    for i in range(m):

        ## finding the maximum in the column and swapping it with the current row (so basically if a zero we move to next column)
        while j<n and abs(A[np.abs(A[i:, j]).argmax() + i][j]) < TOL:
          A[i:,j]=0.0
          j+=1

        if j == n:
          break
        idx = np.abs(A[i:, j]).argmax() + i
        ## swapping
        A[[i,idx]] = A[[idx,i]]
        E[[i,idx]] = E[[idx,i]]
        if A[i][j] == 0:
          continue

        ## making the pivot column to 1
        E[i] = E[i]/A[i][j]
        A[i] = A[i]/A[i][j]
        
        for k in range(i+1,m):
          E[k] = E[k] - E[i] * A[k][j]  
          A[k] = A[k] - A[i] * A[k][j]
        
        pivot_cols.append(j)
        j+=1

    for idx,col in enumerate(pivot_cols):
      for k in range(0,idx):
          E[k] = E[k] - E[idx] * A[k][col]
          A[k] = A[k] - A[idx] * A[k][col] 

    return A,E,pivot_cols,


def col_space(A_orig):
    """Pivot columns of the ORIGINAL A (not the RREF).
       Returns: matrix whose columns form the column space basis."""
    A = A_orig.copy()
    _,_,pivot_cols = rref(A)
    return A_orig[:,pivot_cols]

def row_space(A_orig):
    """Non-zero rows of RREF.Returns: matrix whose rows are the row space basis."""
    A = A_orig.copy()
    R,_,pivot_cols = rref(A)
    r = len(pivot_cols)
    return R[:r]

def rank(A):
    """Count pivot columns from RREF."""
    _,_,pivot_cols = rref(A)
    return len(pivot_cols)

def nullspace(A_orig):
    """From RREF: identify free variables, construct basis vectors for N(A).
       Each free variable → one basis vector.
       Returns: matrix whose columns are the nullspace basis."""
    A = A_orig.copy()
    R,_,pivot_cols = rref(A)
    n = A.shape[1]
    free_cols = [i for i in range(n) if i not in pivot_cols]
    basis = []
    
    for free in free_cols:

      x = np.zeros(n)
      # giving 1 to free variable and solving for the pivot variables
      x[free] = 1 
      for row,pivot in enumerate(pivot_cols):
        x[pivot] = - R[row][free]

      basis.append(x)
    if len(basis) == 0:
      basis = np.zeros((0,n))
    return np.array(basis)

def left_nullspace(A):
    
    return nullspace(A.T)

def four_subspaces(A):

       cs,ns,rs,l_ns =  col_space(A) , nullspace(A), row_space(A), left_nullspace(A)
       n = A.shape[1]
       d = {"C(A)":{"dim":cs.shape[1],"basis":cs}}
       d['N(A)'] = {"dim":ns.shape[0],"basis":ns}
       d['C(At)'] = {"dim":rs.shape[0],"basis":rs}
       d['N(At)'] = {"dim":l_ns.shape[0],"basis":l_ns}
       return d

def complete_solution(A_orig, b):
    """Returns: particular solution x_p and nullspace basis.
       Full solution set: x = x_p + any combination of nullspace vectors."""
    A = A_orig.copy()
    R,_,pivot_cols = rref(A)
    r = len(pivot_cols)
    ns = nullspace(A)
    b = E @ b
    m,n = A.shape
    x_p = np.zeros(n)
    for row,col in enumerate(pivot_cols):
      x_p[col] = b[row]

    ## if rows beneath r th row isnt 0 in both then say no soln
    for i in range(r,m):
      if abs(b[i]) > TOL:
        # print(b[i],i)
        return None,None

    return x_p , ns


## tests 

for _ in range(1000):
    m, n, r = 5, 4, 3
    A = (np.random.randint(-3, 4, (m, r)) @ np.random.randint(-3, 4, (r, n))).astype(float)

    R, E, piv = rref(A)
    S, s_piv = sympy.Matrix(A.astype(int)).rref()

    assert np.allclose(R, np.array(S, dtype=float)), "RREF wrong"
    assert list(piv) == list(s_piv), "pivots wrong"
    assert rank(A) == np.linalg.matrix_rank(A), "rank wrong"
    assert np.allclose(E @ A, R), "E wrong"

    N = nullspace(A).T
    assert N.shape[1] == null_space(A).shape[1], "nullspace dim wrong"
    assert np.allclose(A @ N, 0), "nullspace vectors wrong"

    x = np.random.randn(n)
    xp, _ = complete_solution(A, A @ x)
    assert np.allclose(A @ xp, A @ x)