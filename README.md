## Regression algorithms

### Generalized Least Squares (gls.m)

MATLAB script that performs generalized least squares for solving the linear regression problem with normally distributed data errors with arbitrart covariance matrix, Σ:

  y = Ax + ε ,   ε ~ N(0,Σ)

If no data covariance matrix is provided, the code performs ordinary least squares (i.e., Σ = Ι). The solution is given by the ordinary least squares solution to the normalized problem,

  y' = A'x + ε' ,   ε ~ N(0,I)

where y' = Wy, A' = WA, ε' = Wε, with W being the inverse square root of Σ, computed using the Cholesky factorization.

The code also computes the covariance matrix of the estimated model parameters, the χ^2 statistic, and the associated P-value.

### Logistic Regression (logistic.m)

