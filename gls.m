function S = gls(A,y,Sig)
% S = gls(A,y,S)
%
% Generalized least squares for solving the linear regression problem 
%   Ax = y + e ,   e ~ N(0,Sig)
% If no data covariance matrix is provided, the code performs ordinary least
% squares.
%
% Inputs:
%   A = design matrix (m x n)
%   y = data (m x 1)
%   Sig = data covariance matrix (m x m)
%
% Outputs:
%   S = structure with solution, containing the following fields:
%       S.xhat = model estimate
%       S.C = model covariance matrix
%       S.yhat = predicted data
%       S.res = residual
%       S.wres = weighted residual
%       S.chi2 = chi-square statistic
%       S.chi2red = reduced chi-square statistic
%       S.Pval = P-value
%
% Andreas Mavrommatis, 2014.

if size(y,1)==1
    y = y';
end

if nargin < 3
    Sig = eye(length(y));
end

L = chol(Sig,'lower'); % Cholesky factorization of data covariance matrix

% Weight by inverse square root of data covariance matrix
At = L\A;
yt = L\y;

% Ordinary least squares on weighted problem
xhat = At\yt;

% Model covariance matrix
C = inv(At'*At);

% Predicted data, residual, and weighted residual
yhat = A*xhat;
res  = y - yhat;
wres = L\res;

% Chi-square statistic
chi2 = wres'*wres;
m = length(y);  % number of data
n = length(xhat); % number of model parameters
chi2red = chi2/(m-n); % reduced chi-square

% P-value
Pval = 1 - chi2cdf(chi2,n);

% Parse into ouput structure
S.xhat = xhat;
S.C = C;
S.yhat = yhat;
S.res = res;
S.wres = wres;
S.chi2 = chi2;
S.chi2red = chi2red;
S.Pval = Pval;

end
