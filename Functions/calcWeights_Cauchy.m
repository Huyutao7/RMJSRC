function [ w ] = calcWeights_Cauchy( res )

%Update tau-----------------------------------------------------
e2     =   res.^2; tau2=0.5*mean(e2)+eps; %1 for clean, 0.5 or even less for noisy, default 1
%Update weight----------------------------------------------------- 
w = 1./(ones(size(e2))+e2/tau2);    
% w=sqrt(w);
end

