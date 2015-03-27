function [xhat, resnorm, t] = irls(y,A,x0,tol)
% function [xhat, resnorm, t] = irls(y,A,x0,tol)
%
% Compute the IRLS (Iterative Reweighted Least Squares) solution to the L1
% minimization problem min ||y - A*x||_1
% Inputs:   y = data vector
%           A = design matrix
%           x0 = starting model, typically the L2 solution
%           tol = tolerance for convergence
% Outputs:  xhat = final IRLS estimate
%           resnorm = 1-norm of the residual for each iteration
%           t = relative model error for each iteration
%
% Andreas Mavrommatis, 2012


% First step
r{1} = y - A*x0;                % Residual vector
R{1} = diag(1./abs(r{1}));      % Weigthing matrix
x{1} = (A'*R{1}*A)\(A'*R{1}*y); % Current estimate
t{1} = inf;                     % Tolerance (inf, so we always start)
resnorm{1} = norm(r{1},1);      % 1-norm of residual
t{2} = inf;                     % Update tolerance
k = 2;                          % Update count

% Repeat until convergence - make sure we don't divide by zero
while  t{k-1} > tol && all(abs(r{k-1}) >= eps)
    r{k} = y - A*x{k-1};
    R{k} = diag(1./abs(r{k}));
    x{k} = (A'*R{k}*A)\(A'*R{k}*y);
    t{k} = norm(x{k} - x{k-1})/(1 + norm(x{k}));
    resnorm{k} = norm(r{k},1);   
    k = k + 1;
end

xhat = x{end};

