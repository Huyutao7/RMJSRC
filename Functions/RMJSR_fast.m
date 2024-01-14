function [ w,var ] = RMJSR_fast(pars,b)

T = pars.T;
M=length(T);
n = size(T{1},2);
x = repmat(ones(n,1)/n,[1,M]);
z = zeros(n,M);
u2 = zeros(n,M);

e=cell(M,1);u1=cell(M,1);w=cell(M,1);
residual_init=cell(M,1);residual=cell(M,1);
for j=1:M
    e{j} = zeros(size(b{j}));
    u1{j}=zeros(size(b{j}));
    residual_init{j}=b{j} - normc(mean(T{j},2)); 
    w{j}= calcWeights_Cauchy(residual_init{j});
    residual{j}=zeros(size(b{j}));
end


var.x = x;
var.z = z;
var.e = e;
var.u1 = u1;
var.u2 = u2; 

weight_prev = w;

for t=1:pars.kappa
    
%     var = firc_admm(T,b,w,pars,var,pars.alg);
    var = RMJSR_admm(T,b,w,pars,var);
    weight_g=0;
    for j=1:M
        residual{j} = b{j} - T{j}*var.x(:,j);
%         w{j} = calcWeights(residual{j});
        w{j} = calcWeights_Cauchy(residual{j});
        weight_g  =weight_g+ norm(w{j}-weight_prev{j},2)/norm(weight_prev{j},2);
%         weight_g = weight_g+norm(w{i}-weight_pref{i},2)/norm(weight_pref{i},2);  
    end

    weight_prev = w;
    
    if weight_g < pars.epsilon_3
        break;
    end
end

for j=1:M
    w{j}=sqrt(w{j});
end
end

