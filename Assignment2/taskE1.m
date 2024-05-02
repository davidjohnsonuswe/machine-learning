%% load data
close all
load 'A2_data.mat'

%% Compute a linear PCA and vizualize the whole training data in d = 2 dimensions

X = train_data_01;
proj_data = linear_pca(X);

%% Plot
gscatter(proj_data(1, :), proj_data(2, :), train_labels_01, 'rb', 'o*');
legend('Class 1', 'Class 2');
xlabel('First principal component');
ylabel('Second principal component');
%print -depsc E1.eps