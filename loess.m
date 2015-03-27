function yhat = loess(x,y,tau)
% yhat = loess(x,y,tau)
% 
% Perform locally weighted regression using a Gaussian weight kernel.
%
% Inputs:
%   x = (m x 1) vector of values of independent variable
%   y = (m x 1) vector of observed values of response variable 
%   tau = bandwidth for the Gaussian weight function
% 
% Output:
%   yhat = predicted response
%
% Andreas Mavrommatis, 2014.

m = size(x,1);      % number of training examples

% Design matrix -- add intercept term
X = [ones(m,1) x];

yhat = nan(m,1);
for i = 1:m

    % Weight matrix
    W = locweightmat(x,x(i),tau);

    % weighted least squares solution
    theta = (X'*W*X)\(X'*W*y);

    % predicted data
    yhat(i) = X(i,:)*theta;

end

function W = locweightmat(x,xq,tau)
% computes diagonal weight matrix for locally weighted regression

m = size(x,1);      % number of training examples
W = eye(m);

for i = 1:m
    W(i,i) = exp(- ((xq - x(i))^2)/(2*tau^2) );
end
