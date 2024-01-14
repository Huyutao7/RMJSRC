%RMJSRC_fast_MulPIE_Occ
close all;clear;clc;
addpath ..\Data ..\Functions

method='RMJSRC_fast';

%load data
load MulPIE_32x32_trtt_Occ60
num_modal=length(Tr_dat); Tt_dat_Occ=cell(num_modal,1);
%-------------------------------------------------------------------------

%-------------------------------------------------------------------------
%parameter setting
K=68;% num of classes
lambda=1e-3; %well tuned

%construct training and test data
for i=1:num_modal
    tr_dat=Tr_dat{i};tt_dat=Tt_dat{i};
    Ind=find(trls<=K);tr_dat=tr_dat(:,Ind);trls=trls(:,Ind);
    Ind2=find(ttls<=K);tt_dat=tt_dat(:,Ind2);ttls=ttls(:,Ind2);
    tr_dat=NormalizeFea(tr_dat,0);
    tt_dat_Occ=NormalizeFea(tt_dat,0);
    Tr_dat{i}=tr_dat;Tt_dat_Occ{i}=tt_dat_Occ;
end
%-------------------------------------------------------------------------
%perform classification
tic;
ID=RMJSRC_fast(Tr_dat,Tt_dat_Occ,trls,lambda);
time=toc;
cornum      =   sum(ID==ttls);
Rec         =   cornum/length(ttls)*100;
fprintf(['recogniton rate: ' num2str(Rec)]);fprintf('\n');
fprintf(['time: ' num2str(time)]);fprintf('\n');


