## Regression algorithms

### `gls.m`, `gls.py`: Generalized Least Squares

Linear regression using [generalized least squares](http://en.wikipedia.org/wiki/Generalized_least_squares) and assuming normally distributed data errors according to an arbitrary covariance matrix, Σ:

  y = Ax + ε ,   ε ~ N(0,Σ)

If no data covariance matrix is provided, the code performs ordinary least squares (i.e., assumes Σ = Ι). Generalized linear squaressolves the problem of minimizing the squared L2 norm of the weighted residual,

  min ||W(y - Ax)||_2^2

where the weighting matrix W is the inverse square root of Σ, computed using the [Cholesky factorization](http://en.wikipedia.org/wiki/Cholesky_decomposition). The code also computes the covariance matrix of the estimated model parameters, the χ^2 statistic, and the associated P-value.

### `irls.m`: Robust (L1) regression using Iterative Reweighted Least Squares

Robust linear regression by minimizing the L1 norm of the data residual,

  min ||y - Ax||_1

using an [iterative reweighted least squares](http://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares) algorithm. L1 regression is less sensitive to outliers than standard least squares.

### `lasso_irls`: Lasso regularization using IRLS

[Lasso regularized regression](http://en.wikipedia.org/wiki/Least_squares#Lasso_method) (aka sparsity-based regularization) by solving the problem

  min ||y - Ax||_2^2 + λ||x||_1

where λ is a tunable scalar that is chosen based on a user-input tolerance on the data residual. The code uses an iterative reweighted least squares (ISLR) method to converge to the final estimate for x.

### `loess.m`: Local regression

[Local regression](http://en.wikipedia.org/wiki/Local_regression) using locally weighted least squares and a Gaussian weighting kernel.


### Coming soon:

* SVD regularization
* Feasible generalized least squares (using a correlation model)
