function [phi, Delta_phi] = computePRC(t_spikes, t_pulses)
% 
% [phi, Delta_phi] = computePRC(t_spikes, t_pulses)
%
% Traditional direct method – without normalization 
%
% t_spikes  :  1 x N   array, containing the times of the recorded spikes
% t_pulses  :  1 x M   array, containing the times of the delivered pulses
%
% Michele Giugliano - michele.giugliano@uantwerpen.be
% Joao Couto - jpcouto@gmail.com
%

ISIm      = mean(diff(t_spikes));
M         = length(t_pulses);
phi       = zeros(M,1);
Delta_phi = zeros(M,1);

for k=1:M
	idx_preceding = find(t_spikes < t_pulses(k), 1, 'Last');
        idx_following = idx_preceding+1;
        tau           = t_pulses(k) - t_spikes(idx_preceding);    
        Ti            = t_spikes(idx_following) - t_spikes(idx_preceding);
        phi(k)        = tau / ISIm;
        Delta_phi(k)  = 1. - Ti / ISIm;
end

