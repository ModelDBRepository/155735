function [phi, Delta_phi] = computePRCcorrected(t_spikes, t_pulses)
%
% [phi, Delta_phi] = computePRCcorrected(t_spikes, t_pulses)
%
% Corrected direct method (Phoka et al., 2010) – without normalization
% t_spikes  :  1 x N   array, containing the times of the recorded spikes
% t_pulses  :  1 x M   array, containing the times of the delivered pulses
%
% Michele Giugliano - michele.giugliano@uantwerpen.be
% Joao Couto - jpcouto@gmail.com
%

ISIm      = mean(diff(t_spikes));
M         = length(t_pulses);
phi       = zeros(M,3);
Delta_phi = zeros(M,3);

for k=1:M
    idx_preceding = find(t_spikes < t_pulses(k), 1, 'Last');
    idx_following = idx_preceding+1;
    tau           = t_pulses(k) - t_spikes(idx_preceding);       
    Ti            = t_spikes(idx_following) - t_spikes(idx_preceding);
    Tim1          = t_spikes(idx_preceding) - t_spikes(idx_preceding-1);
    Tip1          = t_spikes(idx_following+1) - t_spikes(idx_following);
    phi(k,1)      = tau / ISIm;
    phi(k,2)      = (Tim1 + tau) / ISIm;
    phi(k,3)      = (tau - Ti)   / ISIm;
    Delta_phi(k,1)= 1. - Ti / ISIm;
    Delta_phi(k,2)= 1. - Tim1 / ISIm;
    Delta_phi(k,3)= 1. - Tip1 / ISIm;
end

phi       = phi(:);
Delta_phi = Delta_phi(:);
