function mhat = lasso_irls(d,G,m0,tolcon,tolres,a,doplot)
% function mhat = lasso_irls(d,G,m0,tolcon,tolres,a,doplot)
%
% Performs lasso regression using an Iterative Reweighted Least Squares (IRLS)
% method. 
%
% Inputs:   d = data vector
%           G = kernel matrix
%           m0 = starting model 
%           tolcon = tolerance for convergence
%           tolres = tolerance for data residual
%           a = vector of values of smoothing parameter
%           doplot = 1 for plots of the trade-off curves
% Output:  mhat = final estimate
%
% Andreas Mavrommatis, 2012


%  ------------------------- First step -----------------------------------

k = 1;
t{k} = inf;          % Initial tolerance (inf, so we always start)

% Fit within tolerance
[mnorm,rnorm] = deal(zeros(length(a),1));
mtemp = cell(length(a),1);
for i = 1:length(a)
    sqrtW = diag(1./sqrt(abs(m0))); % Square-root of diagonal weight matrix
    mtemp{i} =  [G; (a(i)/sqrt(2))*sqrtW]\[d; zeros(length(m0),1)]; % regularized estimate
    mnorm(i) = norm(mtemp{i},1);     % 1-norm of model vector
    rnorm(i) = norm(d - G*mtemp{i}); % 2-norm of residual
end
iopt = find(rnorm > tolres, 1);
m{1} = mtemp{iopt};  % Current optimal estimate

% Plot trade off curve?
if doplot
    figure; 
    loglog(mnorm,rnorm,'-k','linewidth',2)
    xlabel('||m||_1', 'fontsize', 14)
    ylabel('||r||_2', 'fontsize', 14)
    set(gca,'fontsize',14)
    hold on; box on; axis tight; grid on
    ax = axis;
    loglog([ax(1) ax(2)], [tolres tolres], '--b');
    loglog(mnorm(iopt),rnorm(iopt), 'or')        
    title(['Trade-off curve for iteration ', num2str(k)])
end

k = k+1;
t{k} = inf;          % Tolerance (inf, so we always start)


%  ------------------- Repeat until convergence -----------------------------

while  t{k-1} > tolcon && all(abs(m{k-1}) >= eps) % make sure we don't divide by zero

    % Fit within tolerance
    [mnorm,rnorm] = deal(zeros(length(a),1));
    mtemp = cell(length(a),1);
    for i = 1:length(a)
        sqrtW = diag(1./sqrt(abs(m{k-1}))); % Square-root of diagonal weight matrix
        mtemp{i} =  [G; (a(i)/sqrt(2))*sqrtW]\[d; zeros(length(m0),1)]; % regularized estimate
        mnorm(i) = norm(mtemp{i},1);
        rnorm(i) = norm(d - G*mtemp{i});
    end
    iopt = find(rnorm > tolres, 1);
    m{k} = mtemp{iopt};  % Current optimal estimate

    % Plot trade off curve?
    if doplot
        figure; 
        loglog(mnorm,rnorm,'-k','linewidth',2)
        xlabel('||m||_1', 'fontsize', 14)
        ylabel('||r||_2', 'fontsize', 14)
        set(gca,'fontsize',14)
        hold on; box on; axis tight; grid on
        ax = axis;
        loglog([ax(1) ax(2)], [tolres tolres], '--b');
        loglog(mnorm(iopt),rnorm(iopt), 'or')
        title(['Trade-off curve for iteration ', num2str(k)])
    end

    t{k} = norm(m{k} - m{k-1})/(1 + norm(m{k}));
    k = k + 1;
  
end

mhat = m{end};  % Final, optimal estimate

end

