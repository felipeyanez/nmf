function [W, H, obj, time] = nmf_kl_fpa(V, W, H, N, D)

% [W, H, obj, time] = nmf_kl_fpa(V, W, H, N, D)
%
%
% Non-negative matrix factorization (NMF) implementation using a 
% first-order primal-dual algorithm (FPA).
%
% Given a non-negative matrix V, find non-negative matrix factors W and H 
% such that V approx. W*H, i.e. solving the following optimization problem:
%
% min_{W,H} D(V||W*H),
%
% where D(V||W*H) is the Kullback-Leibler divergence loss.
%
% The FPA estimates W and H at each iteration, solving the following:
%
% min_x F(K*x) + G(x),
%
% where K is a known matrix, F(u) = a'*(log(u./a) + 1) and G(u) = sum(K*u).
%
%
% Required Parameters:
%     V:                non-negative given matrix (n x m)
%     W:                initial non-negative matrix factor (n x r)
%     H:                initial non-negative matrix factor (r x m)
%     N:                number of iterations (access to data)
%     D:                number of iterations for each ND problem
% 
% Output:
%     W:                optimal non-negative matrix factor
%     H:                optimal non-negative matrix factor
%     obj:              objective at each iteration (access to data)
%     time:             run time per iteration (access to data)
%
%
% Author: Felipe Yanez, May 2014

% Initialization
t0   = cputime;
obj  = zeros(1,N/D);
time = zeros(1,N/D);

% Set parameters
chi   = -V./(W*H);
chi   = bsxfun(@times, chi, 1./max(bsxfun(@times, -W'*chi, 1./sum(W,1)')));
Wbar  = W;
Wold  = W;
Hbar  = H;
Hold  = H;
[n m] = size(V);
r     = size(H,1);

for i = 1:N/D,
    
    % Computation of H
    sigma = sqrt(n/r) * sum(W(:)) ./ sum(V,1)  / norm(W);
    tau   = sqrt(r/n) * sum(V,1)  ./ sum(W(:)) / norm(W);

    for j = 1:D,
        
        chi  = chi + bsxfun(@times, W*Hbar, sigma);
        chi  = (chi - sqrt(chi.^2 + bsxfun(@times, V, 4*sigma)))/2;
        H    = max(H - bsxfun(@times, W'*(chi + 1), tau), 0);
        Hbar = 2*H - Hold;
        Hold = H;
        
    end
    
    % Computation of W
    sigma = sqrt(m/r) * sum(H(:)) ./ sum(V,2)  / norm(H);
    tau   = sqrt(r/m) * sum(V,2)  ./ sum(H(:)) / norm(H);

    for j = 1:D,
        
        chi  = chi + bsxfun(@times, Wbar*H, sigma);
        chi  = (chi - sqrt(chi.^2 + bsxfun(@times, V, 4*sigma)))/2;
        W    = max(W - bsxfun(@times, (chi + 1)*H', tau), 0);
        Wbar = 2*W - Wold;
        Wold = W;
        
    end
    
    % Objective and run time per iteration
    obj(i)  = sum(sum(-V.*(log((W*H+eps)./(V+eps))+1)+W*H));
    time(i) = cputime - t0;
        
end

obj  = repmat(obj, D,1);
time = repmat(time,D,1);

obj  = obj(:)';
time = time(:)';

end