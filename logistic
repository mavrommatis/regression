function [theta,h,g,H,Niter] = logistic(x,y,theta0,tol)
% [theta,h,g,H,Niter] = logistic(x,y,theta0,tol)
%
% Performs logistic regression in a set of (x,y) measurements, where x is
% (m x n) and y is (m x 1) and takes the values {0,1}.
%
% Inputs:
%   x = (m x n) matrix, where m = number of training examples (data) and n
%       is number of features
%   y = (m x 1) vector of either 0's or 1's
%   theta0 = (n x 1) vector of initial guess on unknown parameters
%   tol = tolerance for convergence
%
% Outputs:
%  theta = (n x 1) vector of final estimate of unknown parameters
%  h = (m x 1) vector of predicted values of logistic function
%  g = (n x 1) gradient vector at the final solution
%  H = (n x n) Hessian matrix at the final solution
%  Niter = number of iterations needed for convergence.
%
% Andreas Mavrommatis, 2014.


% initialization
theta = theta0;  
dtheta = Inf;
Niter = 0;

% repeat until convergence
while dtheta > tol
    
    % compute gradient vector of likelihood function
    g = gradlik(x,y,theta);
    
    % compute Hessian matrix of likelihood function
    H = hesslik(x,theta);
    
    % perform Newton step
    theta_new = theta - H\g;
    
    % compute difference between successive estimates
    dtheta = norm(theta_new - theta);
    
    % update estimate
    theta = theta_new;
    Niter = Niter + 1;
    
end

% compute predicted value of logistic function for each training set
m = size(x,1);  
h = nan(m,1);
for i = 1:m
    h(i) = logistic_fun(theta'*x(i,:)');
end


end



function g = gradlik(x,y,theta)
% computes the gradient vector of the likelihood function

m = size(x,1);  % number of training examples
n = size(x,2);  % number of features

g = zeros(n,1);
for j = 1:n
    for i = 1:m
        h = logistic_fun(theta'*x(i,:)');
        g(j) = g(j) + (y(i) - h)*x(i,j);        
    end
end

end


function H = hesslik(x,theta)
% computes the Hessian matrix of the likelihood function

m = size(x,1);  % number of training examples
n = size(x,2);  % number of features

H = zeros(n,n);
for j = 1:n
    for k = 1:n
        for i = 1:m
            h = logistic_fun(theta'*x(i,:)');
            H(j,k) = H(j,k) - h*(1-h)*x(i,k)*x(i,j);
        end
    end
end

end


function h = logistic_fun(z)

h = 1./(1 + exp(-z));

end

