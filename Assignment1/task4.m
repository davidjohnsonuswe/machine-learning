%% Task 4.1 Produce reconstruction plots

close all
% load data
load A1_data.mat

% Case 1
lambda = 0.1;
what1 = lasso_ccd(t,X,lambda);
y = X*what1;
yhat = Xinterp*what1;

figure(1)
hold on
plot(n,t,'mo')
plot(n,y,'b*')
plot(ninterp,yhat,'r')
legend('Original Data','Reconstructed Data','Interpolated Reconstruction')
xlabel('Time');
%print -depsc case1.eps

% Case 2
lambda = 10;
what2 = lasso_ccd(t, X, lambda);
y = X*what2;
yhat = Xinterp*what2;

figure(2)
hold on
plot(n,t,'mo')
plot(n,y,'b*')
plot(ninterp,yhat,'r')
legend('Original Data','Reconstructed Data','Interpolated Reconstruction')
xlabel('Time');
%print -depsc case2.eps

% Case 3
lambda = 1.5;
what3 = lasso_ccd(t,X,lambda);
y = X*what3;
yhat = Xinterp*what3;

figure(3)
hold on
plot(n,t,'mo')
plot(n,y,'b*')
plot(ninterp,yhat,'r')
legend('Original Data', 'Reconstructed Data', 'Interpolated Reconstruction')
xlabel('Time');
%print -depsc case3.eps

%% Task 4.2

% Count the number of non-zero coordinates
non_zero_co1 = sum(what1~=0);
non_zero_co2 = sum(what2~=0);
non_zero_co3 = sum(what3~=0);