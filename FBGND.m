function [X,N,S,Out] = FBGND(Y,opts)

%%  Parameters setting
Nway = size(Y);
tol = 1e-3;
max_iter = 200;
Llevel=opts.Llevel;
Nlevel=opts.Nlevel;
lambda1=opts.beta(5)*(opts.level*Llevel)^2;
lambda2=(opts.beta(6)*(opts.level*Nlevel)^2);
lambda3 = opts.lambda3;
lambda4 = opts.lambda4;
gamma=opts.gamma;
beta = opts.beta;
constant_M = 2*1e10;
rank=opts.rank;
global sigmas
useGPU=1;

%% Initialization
L = Y;
M = Y;
X = zeros(Nway);
S = zeros(Nway);
N = zeros(Nway);
F1 = Y;
F2 = Y;
F3 = Y;
P1 = zeros(Nway);
P2 = zeros(Nway);
P3 = zeros(Nway);
P4 = zeros(Nway);
P5 = zeros(Nway);
P6 = zeros(Nway);
Out.Res=[]; Out.PSNR=[];

%% Import FFDNet

load('FFDNet_color.mat');
net = vl_simplenn_tidy(net);
if useGPU
    net = vl_simplenn_move(net, 'gpu');
end

%%
for iter = 1 : max_iter
   Xold=X;
    %% Update X
    temp = beta(1)*F1-P1 + beta(2)*F2-P2 + beta(3)*F3-P3 + beta(4)*(Y-N-S)+P4+beta(5)*L-P5+beta(6)*M-P6;
    X = temp/sum(beta);
    
   %% Update F 
    X1 = permute(X,[2,3,1]);  X2 = permute(X,[3,1,2]);  X3 = X;
    p1 = permute(P1,[2,3,1]); p2 = permute(P2,[3,1,2]); p3 = P3;
    F1=rtsvd(X1+p1/beta(1),rank(1),30);
    F2=rtsvd(X2+p2/beta(2),rank(2),30);
    [n,~,~]=size(X3);
    if rank(3)+50>=n
        F3=rtsvd(X3+p3/beta(3),rank(3),0);
    else
        F3=rtsvd(X3+p3/beta(3),rank(3),50);
    end
    F1 = ipermute(F1,[2,3,1]);
    F2 = ipermute(F2,[3,1,2]);
     % supplement
    Delta1=X-F1+P1/beta(1);
    if norm(Delta1(:))>sqrt(constant_M/beta(1))
        F1=X+P1/beta(1);
    end
    Delta2=X-F2+P2/beta(2);
    if norm(Delta2(:))>sqrt(constant_M/beta(2))
        F2=X+P2/beta(2);
    end
    Delta3=X-F3+P3/beta(3);
    if norm(Delta3(:))>sqrt(constant_M/beta(3))
        F3=X+P3/beta(3);
    end
    %% Update L
    sigma1=sqrt(lambda1/beta(5));
    if sigma1>=0.005 % mute FFDNet
        input1 =X + P5/beta(5);
        input1 = single(input1); 
        if mod(Nway(3),3)==1
            input1=cat(3,input1,input1(:,:,end-1:end));
        elseif mod(Nway(3),3)==2
            input1=cat(3,input1,input1(:,:,end));
        end

        if mod(Nway(1),2)==1
            input1 = cat(1,input1, input1(end,:,:)) ;
        end
        if mod(Nway(2),2)==1
            input1 = cat(2,input1, input1(:,end,:)) ;
        end
        [~,~,n3]=size(input1);

        if useGPU
            input1 = gpuArray(input1);
        end
        
        % translate
        input2 = cat(3,input1(:,:,2:end),input1(:,:,1));
        input3 = cat(3,input1(:,:,3:end),input1(:,:,1:2));

        sigmas =sigma1;
        for i=1:n3/3   
            res = vl_simplenn(net,input1(:,:,3*(i-1)+1:3*(i-1)+3),[],[],'conserveMemory',true,'mode','test');
            output1(:,:,3*(i-1)+1:3*(i-1)+3)=res(end).x;
            res = vl_simplenn(net,input2(:,:,3*(i-1)+1:3*(i-1)+3),[],[],'conserveMemory',true,'mode','test');
            output2(:,:,3*(i-1)+1:3*(i-1)+3)=res(end).x;
            res = vl_simplenn(net,input3(:,:,3*(i-1)+1:3*(i-1)+3),[],[],'conserveMemory',true,'mode','test');
            output3(:,:,3*(i-1)+1:3*(i-1)+3)=res(end).x;
        end  

        % inver translate
        output2=cat(3,output2(:,:,end),output2(:,:,1:end-1));
        output3=cat(3,output3(:,:,end-1:end),output3(:,:,1:end-2));
        output=(output1+output2+output3)/3;

        if mod(Nway(3),3)==1
            output=output(:,:,1:end-2);
        elseif mod(Nway(3),3)==2
            output=output(:,:,1:end-1);
        end

        if mod(Nway(1),2)==1
            output = output(1:end-1,:,:);
        end
        if mod(Nway(2),2)==1
            output = output(:,1:end-1,:);
        end
        if useGPU
            L = double(gather(output)); 
        else
            L = double(output);
        end
    else
        L=X + P5/beta(5);
    end
    %% Update M
    sigma2=sqrt(lambda2/beta(6));
    if sigma2>0.005 % mute BM4D 
        if opts.speedup==1
            M=X;
        else
            temp =X+P6/beta(6);
            [M,~]=bm4d(temp, 'Gauss', sigma2,'np', 1, 0);
        end
    elseif (sigma2>=0.002 && sigma2<=0.005)   
        temp =X+P6/beta(6);
        [M,~]=bm4d(temp, 'Gauss', sigma2,'np', 1, 0);
    else
        M=X+P6/beta(6);
    end
 
    %% Update N
    N=beta(4)*(Y-X-S+P4/beta(4))/(lambda3+beta(4));
    
    %% Update S
    S = prox_l1(Y-X-N+P4/beta(4),lambda4/beta(4)); 
       
    %% Check the convergence
    chg=norm(Xold(:)-X(:))/norm(Xold(:));
    Out.Res = [Out.Res,chg];
    
    % Save parameters        
    if isfield(opts, 'Xtrue')
        XT=opts.Xtrue;
        psnr = PSNR3D(XT * 255, X * 255);
        Out.PSNR = [Out.PSNR,psnr];
    end
    
    if iter==1 || mod(iter, 10) == 0
        if isfield(opts, 'Xtrue')
       fprintf('FBGND: iter = %d   PSNR= %f   res= %f \n',...
       iter, psnr, chg);
        else
            fprintf('FBGND: iter = %d   res= %f \n', iter, chg);
        end
    end
    
    if chg < tol
        break;
    end
    
    %% Update multipliers & penalty parameters
    P1 = P1 + beta(1)*(X-F1);
    P2 = P2 +beta(2)*(X-F2);
    P3 = P3 + beta(3)*(X-F3);
    P4=P4+beta(4)*(Y-X-N-S);
    P5=P5+beta(5)*(X-L);
    P6=P6+beta(6)*(X-M);
    beta = gamma.*beta;
    imshow(X(:,:,45));
    drawnow;
end
end