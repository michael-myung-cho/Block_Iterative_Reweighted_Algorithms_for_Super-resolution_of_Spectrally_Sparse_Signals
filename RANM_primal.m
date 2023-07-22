function [EDFreq,itr] = RANM_primal(YonOmega, Omega, N, k, fplot, freq, ck)
%% Reweighted Atoimc Norm Minimization (RANM) in Primal form
% (NOTE) The frame of this code is obtained from Z. Yang. (https://sites.google.com/site/zaiyang0248/) 
% For displaying the results, few more codes are added without modifying the main code.
%

% [Y,u,freq,amp] = ANM_sdpt3(YonOmega, Omega, N, eta)
% 
% ANM_sdpt3 implements the atomic norm minimization problem:
% min_Y ||Y||_A, subject to ||Y_Omega - Yo_Omega||_F <= epsilon
% via duality using SDPT3.
% 
% The dual problem is
% min_{V,H} <V_Omega, Yo_Omega>_R + epsilon*||V_Omega||_F,
% subject to [I V'; V H] >= 0, V_Omegac = 0, and T^*(H) = [1,0,...,0]^T.
% 
% Input:
%   YonOmega: observed measurements on Omega
%   Omega: index set of the measurements
%   N: length of sinusoidal signal of interest
% Output:
%   Y: recovered Y
%   u: u composing the Toeplitz matrix
%   freq: recovered frequency
%   amp: recovered amplitude
% 
% References:
% Z. Yang and L. Xie, "Continuous Compressed Sensing With a Single or ...
%     Multiple Measurement Vectors", IEEE Workshop on Statistical Signal ...
%     Processing (SSP), pp. 308--311, June 2014.
% Z. Yang and L. Xie, "Exact joint sparse frequency recovery via ...
%     optimization methods", http://arxiv.org/abs/1405.6585, May 2014.
% 
% Written by Zai Yang, Feb. 2014


Omegac = (1:N)';
Omegac(Omega) = [];


MaxItr = 20;
uOld = zeros(N,1);
epsilon = 1/2;
YOld = zeros(N,1);

for itr = 1:MaxItr
tic;   
if itr <= 10
    epsilon = 1/2^itr;
else
    epsilon = 1/2^10;
end
W = inv(toeplitz(uOld) + epsilon*eye(N));

% solve the dual problem
cvx_quiet true
cvx_precision default
cvx_solver sdpt3

cvx_begin

  variable Y(N) complex;
  variable u(N) complex;
  variable X complex;
  
  minimize real(trace( W*toeplitz(u) ) + trace(X))
  subject to 
        [X Y'; Y toeplitz(u)] == semidefinite(N+1, N+1),
        norm(Y(Omega) - YonOmega(Omega),2) <= 0,
cvx_end

if norm(Y-YOld,2) < 10^-6
    break;
end
YOld = Y;
uOld = u;

toc;
end

% % Y estimate
% Y = U(2:1+N,1) * 2;
% 
% % postprocessing
% u = U(2,2:1+N).';
[fEst, amp] = VanDec(u);

amp = amp * 2;

% display Mean Square Error (MSE)
fprintf('\n\n');
nfEst = max(size(fEst));
fErr = 0;
if nfEst >= k
    freqC = [freq;min(freq)+1;max(freq)-1];
    for i=1:nfEst
    % freq error
    [res, ind] = sort(abs(freqC - fEst(i)),'ascend');
    fErr = fErr + res(1)^2;
    end
else 
    fEstC = [fEst;min(freq)+1;max(freq)-1]
    for i=1:k    
        [res, ~] = sort(abs(freq(i) - fEstC),'ascend');
        fErr = fErr + res(1)^2;
    end
end
EDFreq = sqrt(fErr);
fprintf('RANM: MSE of freq = %f (itr = %d)\n', EDFreq, itr);

end