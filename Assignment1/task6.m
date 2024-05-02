%% Task 6

close all
% load data
load A1_data.mat

N_lambda = 100;
% K-folds
K = 3;
lambda_min = 0.0001;
N_XAudio = size(Xaudio,1);
% Numbers of sample per fold
Nval = floor(length(Ttrain)/N_XAudio);
lambda_max = 0;

% Loop through all the folds to find lambda_max
for idx=1:Nval
    temp = max(abs(Xaudio'*Ttrain(1+N_XAudio*(idx-1):idx*N_XAudio)));
    lambda_max = max(lambda_max,temp);
end

lambda_grid = exp(linspace(log(lambda_min),log(lambda_max),N_lambda));
[wopt,lambdaopt,RMSEval,RMSEest] = multiframe_lasso_cv(Ttrain, Xaudio,lambda_grid,K);

% Plot
figure(1)
hold on
plot(log(lambda_grid),RMSEval,'-mo');
plot(log(lambda_grid), RMSEest, '-b*')
xline(log(lambdaopt), '--r')
legend('RMSEVal', 'RMSEEst', ['Optimal \lambda =',num2str(round(lambdaopt,4),'%g')], 'Location','best')
xlabel('log(\lambda)')
%print -depsc task6_1.eps