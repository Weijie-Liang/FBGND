%% =================================================================
% This script performs the FBGND algorithm for hyperspectral image restoration.
% 
% Detail can be found in [1]:
% [1] Weijie Liang, Zhihui Tu, Jian Lu, Kai Tu, Michael K. Ng, and Chen Xu.
% Fixed-Point Convergence of Multi-block PnP ADMM and Its Application to
% Hyperspectral Image Restoration.
% IEEE Transactions on Computational Imaging. 2024.
%
% !!! Please make sure that Matconvnet is installed correctly. For more details, 
% please visit https://www.vlfeat.org/matconvnet/.
%
% Created by Weijie Liang
% 4/20/2024
%% =================================================================
clear;
clc;
close all;
addpath(genpath(cd));

%% load data
load('Normalized_pavia.mat');
Ohsi = Normalized_pavia;

%% noise adding
Nway=size(Ohsi);
n1=Nway(1);
n2=Nway(2);
n3=Nway(3);

% Gaussian noise 
G_level=0.1;
Noise=G_level.*randn(size(Ohsi));
Nhsi=Ohsi+Noise;

% Salt & Pepper nosie 
S_level=0.25;
for i = 1:n3  
Nhsi(:,:,i)=imnoise(Nhsi(:,:,i),'salt & pepper',S_level);
end

%%
EN_FBGND =  1;
methodname  = { 'Observed','FBGND'};
Mnum = length(methodname);

%% evaluation indexes
Re_hsi  =  cell(Mnum,1);
psnr    =  zeros(Mnum,1);
ssim    =  zeros(Mnum,1);
sam     =  zeros(Mnum,1);
time    =  zeros(Mnum,1);

%% observed
i  = 1;
Re_hsi{i} = Nhsi;
[psnr(i), ssim(i), sam(i)] = MSIQA(Ohsi * 255, Re_hsi{i} * 255);
disp(['performing ',methodname{i}, ' ... ']);
fprintf('noise: PSNR=%5.4f   \n',  psnr(i));
enList = 1;

%% performing FBGND
i=i+1;
if EN_FBGND
    %%%%%%
    opts=[];
    opts.lambda3=50;
    opts.level=G_level;
    opts.lambda4=1;
    opts.gamma=1.2*[1,1,1,1,1,1];
    opts.beta=[0.1,0.1,1e-3,0.4,0.1,0.02];
    opts.rank=[round(min(n2,n3)*0.05),round(min(n1,n3)*0.05),round(min(n1,n2)*0.7)];
    opts.Xtrue=Ohsi;
    opts.Llevel=2.8;
    opts.Nlevel=2;
    opts.speedup=1;
    %%%%%%
    fprintf('\n');
    disp(['performing ',methodname{i}, ' ... ']);
    t0= tic;
    [Re_hsi{i},~,~,Out]=FBGND(Nhsi,opts);
    time(i) = toc(t0);
    [psnr(i), ssim(i), sam(i)] = MSIQA(Ohsi * 255, Re_hsi{i} * 255);
    enList = [enList,i];
    fprintf('FBGND: PSNR=%5.4f   \n',  psnr(i));
end

%% show result
fprintf('\n');
fprintf('================== Result ==================\n');
fprintf('      %5.2s %5.3f      %5.2s %5.3f    \n', 'G:',G_level, 'S:', S_level);
fprintf('================== Result ==================\n');
fprintf(' %8.8s    %5.4s      %5.4s    %5.4s    %5.4s \n', 'method','PSNR', 'SSIM', 'SAM','Time');
for i = 1:length(enList)
    fprintf(' %8.8s    %5.4f    %5.4f    %5.4f    %5.4f   \n',...
        methodname{enList(i)},psnr(enList(i)), ssim(enList(i)), sam(enList(i)),time(enList(i)));
end
fprintf('================== Result ==================\n');
