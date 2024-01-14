function var = RMJSR_admm(A,b,w,pars,var)
%------------------------------------------------------------------------
% MJSR: Multimodal Joint Sparse Representation
%A:cell Vx1, A{i}: d_ixn, ----multi-view dictionary
%b:cell Vx1, b{i}: d_ix1, ----multi-view test sample
%w:cell Vx1, b{i}: d_ix1, ----multi-view weight vector
%pars:epsilon_1, epsilon_2, Pinv, rho1, rho2,lambda  ----parameters
%var:x, e, z,  ----variables
%lambda: a non-negative parameter
%X (Output): nxV
%------------------------------------------------------------------------


M = length(A); %modalnum
% n=size(A{1},2); %samplenum


MAX_ITER = 1000;%1000
RELTOL   = pars.epsilon_1;
RELTOL1   = pars.epsilon_2;
Pinv = pars.Pinv;
lambda=pars.lambda;

x = var.x; %a 
e = var.e;
z = var.z;
u1 = var.u1;
u2 = var.u2;



c=cell(M,1);A_x=cell(M,1); 

for j=1:M
    c{j}= 1 + (2*w{j}/(pars.rho1));
    A_x{j} = A{j}*x(:,j);
end


for k = 1:MAX_ITER
    
    % e-update
    for j=1:M
        e{j} = shrinkageW(b{j} - A_x{j} + (u1{j}/(pars.rho1)),c{j});
    end
        
    
    % z-update
%     z = subplus(x+(u2/pars.rho2));
    z = shrinkL1L2(x+(u2/pars.rho2),lambda/pars.rho2); %Need to revise
    
    
    % x-update
    for j=1:M
        new_bj = (b{j} - e{j} + (u1{j}/(pars.rho1)));
        q = A{j}'*new_bj + (pars.rho2/pars.rho1)*z(:,j) - (u2(:,j)/pars.rho1);
        x(:,j)= Pinv{j}*q;
        A_x{j} = A{j}*x(:,j);
    end

    
    % u-update
    % update Multipliers
    leq1=cell(M,1);leq2=cell(M,1);
    for j=1:M
    leq1{j} = (b{j} - A_x{j} - e{j});
    leq2{j} = (x(:,j) - z(:,j));
    
    
    u1{j} = u1{j} + pars.rho1*(leq1{j});
    u2(:,j)= u2(:,j) + pars.rho2*(leq2{j});
    end
    
    stopC = mv_norm(leq1); %norm for multi-view data
    stopC1 = mv_norm(leq2);
    
    if (stopC<RELTOL && stopC1<RELTOL1)
        break;
    end
end

var.x = x;
var.e = e;
var.z = z;
end


function z = shrinkageW(x,c)
z = x./c;
end

function  norm_value= mv_norm(x)
M=length(x);norm_value=0;
for j=1:M
    norm_value=norm_value+norm(x{j});
end
end

function V=shrinkL1L2(E,lambda)
    n_row=size(E,1); V=zeros(size(E));
    for i=1:n_row
        e=E(i,:);
        V(i,:)=pos(1-lambda/norm(e))*e;
    end
end 