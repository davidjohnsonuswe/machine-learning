%% load data
close all
load 'A2_data.mat'
X = train_data_01;

%% Task E2: K-means with 2 clusters
K = 2;
[y2, C2] = K_means_clustering(X, K);
proj_data = linear_pca(X);

% Plot classes
figure(1)
gscatter(proj_data(1, :), proj_data(2, :), y2, 'rb', 'o*');
legend('Class 1', 'Class 2');
xlabel('First principal component');
ylabel('Second principal component');
%print -depsc E21.eps

%% K-means with 5 clusters
K = 5;
[y5, C5] = K_means_clustering(X, K);

% Plot classes
figure(2)
gscatter(proj_data(1, :), proj_data(2, :), y5, 'rbgck', 'o*+xs');
legend('Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5');
xlabel('First principal component');
ylabel('Second principal component');
print -depsc E22.eps

%% Task E3, Display the K = 2 centroids as images
C2_1 = reshape(C2(:,1),[28 28]);
C2_2 = reshape(C2(:,2),[28 28]);

figure(3)
hold on
subplot(1,2,1);
imshow(C2_1);
title('Cluster 1')
subplot(1,2,2);
imshow(C2_2);
title('Cluster 2')
print -depsc E31.eps

%% Display the K = 5 centroids as images
C5_1 = reshape(C5(:, 1), [28 28]);
C5_2 = reshape(C5(:, 2), [28 28]);
C5_3 = reshape(C5(:, 3), [28 28]);
C5_4 = reshape(C5(:, 4), [28 28]);
C5_5 = reshape(C5(:, 5), [28 28]);

figure(4)
hold on
subplot(1,5,1);
imshow(C5_1);
title('Cluster 1')
subplot(1,5,2);
imshow(C5_2);
title('Cluster 2')
subplot(1,5,3);
imshow(C5_3);
title('Cluster 3')
subplot(1,5,4);
imshow(C5_4);
title('Cluster 4')
subplot(1,5,5);
imshow(C5_5);
title('Cluster 5')
print -depsc E32.eps