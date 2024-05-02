%% Task 5.1 Plot RMSEVal, RMSEEst and optimal lambda

close all
% load data
load A1_data.mat

N_lambda = 100;
% K-folds
K = 10;
lambda_min = 0.01;
lambda_max = max(abs(X'*t));

lambda_grid = exp(linspace(log(lambda_min),log(lambda_max),N_lambda));
[wopt,lambdaopt,RMSEval,RMSEest] = lasso_cv(t,X,lambda_grid,K);

% Plot
figure(1)
hold on
plot(log(lambda_grid),RMSEval,'-mo');
plot(log(lambda_grid), RMSEest, '-b*')
xline(log(lambdaopt), '--r')
legend('RMSEVal', 'RMSEEst', ['Optimal \lambda =',num2str(round(lambdaopt,1),'%g')], 'Location','best')
xlabel('log(\lambda)')
%print -depsc task5_1.eps

%% Task 5.2 A recontruction plot for the optimal lambda

what_opt = lasso_ccd(t, X, lambdaopt);
y_opt = X*what_opt;
yhat_opt = Xinterp*what_opt;


figure(2)
hold on
plot(n,t,'mo')
plot(n,y_opt,'b*')
plot(ninterp,yhat_opt,'r')
legend('Original Data','Reconstructed Data','Interpolated Reconstruction')
xlabel('Time');
title(['Optimal \lambda = ',num2str(round(lambdaopt,1),'%g')])
%print -depsc task5_2.eps