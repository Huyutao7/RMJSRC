function [id]= RMJSRC_fast(D,Y,Dlabels,lambda)
%Robust Multimodal Joint SR
%------------------------------------------------------------------------
%% PARAMETERS OF ALGORITHM 1
% pars.Tor = train_or;
pars.T = D;
% PARAMETERS RMJSR_fast///////////
pars.epsilon_3 = 0.01;
pars.kappa = 100;
% PARAMETERS RMJSR_admm//////////
pars.lambda_star = 0.01; % use 0.01 or 0.05
pars.lambda = lambda;
pars.rho1 = 1;
pars.rho2 = 0.1;
pars.epsilon_1 = 0.01;
pars.epsilon_2 = 0.05;
%%%%%%%%%%%%%%%%%%%%%%%%%%
M=length(pars.T);R=cell(M,1);
for j=1:M
    R{j}=inv(pars.T{j}'*pars.T{j} + eye(size(pars.T{j}'*pars.T{j},2)).*(pars.rho2/pars.rho1));
end
% R = inv(pars.T'*pars.T + eye(size(pars.T'*pars.T,2)).*(pars.rho2/pars.rho1));
pars.Pinv = R;

%% Algorithm 

N=size(Y{1},2);
id=zeros(1,N); 
modalnum=length(Y);
Dlabels=double(Dlabels);classnum=max(Dlabels);
for i=1:N  
    y=cell(modalnum,1);
    for j=1:modalnum
        y{j}=Y{j}(:,i);
    end
[w,var] =  RMJSR_fast(pars,y);

Error=zeros(classnum,modalnum);
for class  =  1:classnum
    for j=1:modalnum
        coef=var.x(:,j); yy=y{j}; DD=D{j};
        coef_c           =   coef (Dlabels == class);
        Dc   =  DD(:,Dlabels==class);
        z1          =   w{j}.*(yy - Dc*coef_c);
        Error(class,j) = z1(:)'*z1(:);
        
    end
end

error=mean(Error,2);
index   =   find(error==min(error));
id(i)   =   index(1);
end
