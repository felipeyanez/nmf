function [W, H, obj, time] = nmf_kl_mua(V, W, H, maxIter)

% [W, H, obj, time] = nmf_kl_mua(V, W, H, maxIter)
%
% Author: Felipe Yanez, Mar. 2014

% Initialization
t0   = cputime;
obj  = zeros(1,maxIter);
time = zeros(1,maxIter);

for iter = 1:maxIter,
    
    % Multiplicative update rule
    W = W.*((V./(W*H+eps))*H')./repmat(sum(H,2)'+eps,size(V,1),1);
    H = H.*(W'*(V./(W*H+eps)))./repmat(sum(W,1)'+eps,1,size(V,2));
    
    % Compute objective function
    obj(iter)  = sum(sum(-V.*(log((W*H+eps)./(V+eps))+1)+W*H));
    time(iter) = cputime-t0;
    
end

end