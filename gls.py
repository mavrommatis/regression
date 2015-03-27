def gls(A,y,Sig):
    # Generalized least squares for solving the linear regression problem 
    #  Ax = y + e ,   e ~ N(0,Sig)
    # If no data covariance matrix is provided, the code performs ordinary least
    # squares.
    #
    # Inputs:
    #   A = design matrix (m x n)
    #   y = data (m x 1)
    #   Sig = data covariance matrix (m x m)
    #
    # Outputs
    #   xhat = model estimate
    
    import numpy as np

    # Cholesky factorization of data covariance matrix
    L = np.linalg.cholesky(Sig)

    # Weight by inverse square root of data covariance matrix
    Linv = np.linalg.inv(L)
    At = np.dot(Linv,A)
    yt = np.dot(Linv,y)
    
    # Ordinary least squares on weighted problem
    xhat = np.linalg.lstsq(At,yt)[0]
    
    # Model covariance matrix
    C = np.linalg.inv(np.dot(np.linalg.transpose(At),At))
    
    return xhat, C