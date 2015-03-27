## Regression algorithms

### Generalized Least Squares (gls.m, gls.py)

Performs linear regression using [generalized least squares](http://en.wikipedia.org/wiki/Generalized_least_squares) and assuming normally distributed data errors according to an arbitrary covariance matrix, Σ:

  y = A*x + ε ,   ε ~ N(0,Σ)

If no data covariance matrix is provided, the code performs ordinary least squares (i.e., assumes Σ = Ι). Generalized linear squaressolves the problem of minimizing the squared L2 norm of the weighted residual,

  min ||W * (y - A*x)||_2^2

where the weighting matrix W is the inverse square root of Σ, computed using the [Cholesky factorization](http://en.wikipedia.org/wiki/Cholesky_decomposition). The code also computes the covariance matrix of the estimated model parameters, the χ^2 statistic, and the associated P-value.

### Robust (L1) regression using Iterative Reweighted Least Squares (irls.m)

Performs robust linear regression by minimizing the L1 norm of the data residual,

  min ||y - A*x||_1

using an [iterative reweighted least squares](http://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares) algorithm. L1 regression is less sensitive to outliers than standard least squares.

### Lasso regularization using IRLS (lasso_irls.m)

Performs [lasso regularized regression](http://en.wikipedia.org/wiki/Least_squares#Lasso_method) (aka sparsity-based regularization) by solving the following problem

  min ||y - Ax||_2^2 + λ||x||_1

where λ is a tunable scalar that is chosen based on a user-input tolerance on the data residual. The code uses an iterative reweighted least squares (ISLR) method to converge to the final estimate for x.

### Coming soon:

* SVD regularization
* Locally weighted linear regression
* Feasible generalized least squares (using a correlation model)
