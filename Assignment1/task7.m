%% Task 7

close all
% load data
load A1_data.mat

% Calculated from task 6
lambdaopt = 0.015;
Ytest = lasso_denoise(Ttest, Xaudio, lambdaopt);
soundsc(Ytest, fs);

%save('denoised_audio','Ytest','fs');
