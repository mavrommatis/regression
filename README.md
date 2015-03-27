## Regression algorithms

### Generalized Least Squares (gls.m, gls.py)

Code for generalized least squares for solving the linear regression problem with normally distributed data errors with arbitrart covariance matrix, Σ:

  y = Ax + ε ,   ε ~ N(0,Σ)

If no data covariance matrix is provided, the code performs ordinary least squares (i.e., Σ = Ι). The solution is given by the ordinary least squares solution to the transformed problem,

  Wy = WAx + Wε ,   Wε ~ N(0,I)

where the weighting matrix W is the inverse square root of Σ, computed using the Cholesky factorization. The code also computes the covariance matrix of the estimated model parameters, the χ^2 statistic, and the associated P-value.

### Coming soon:

* IRLS algorithm for L1 minimization
* Truncated SVD for underdetermined problems
