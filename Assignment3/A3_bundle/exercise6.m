%% Load model
close all
load models\network_trained_with_momentum.mat

%% Plot the kernels the first convolutional layer learns
first_conv_layer = net.layers{1,2};
first_conv_weights = first_conv_layer.params.weights;

for i = 1:16
    subplot(4,4,i);
    imshow(first_conv_weights(:,:,:,i));
    print -depsc E6_kernels.eps
end

%% Plot misclassified images

% Use the code from mnist_starter.m to test
addpath(genpath('./'));
x_test = loadMNISTImages('data/mnist/t10k-images.idx3-ubyte');
z_test = loadMNISTLabels('data/mnist/t10k-labels.idx1-ubyte');
z_test(z_test==0) = 10;
x_test = reshape(x_test, [28, 28, 1, 10000]);

% Evaluate on the test set. There are 10000 images, so it takes some
% time
pred = zeros(numel(z_test),1);
batch = 16;
for i=1:batch:size(z_test)
    idx = i:min(i+batch-1, numel(z_test));
    % note that z_test is only used for the loss and not the prediction
    y = evaluate(net, x_test(:,:,:,idx), z_test(idx));
    [~, p] = max(y{end-1}, [], 1);
    pred(idx) = p;
end

% find the misclassified indices
misclassified_idx = find(pred ~= z_test);

% display the misclassified images
misclassified_images = x_test(:,:,:,misclassified_idx);
misclassified_labels = pred(misclassified_idx);
true_labels = z_test(misclassified_idx);
figure;
for i = 1:9
    subplot(3,3,i);
    imshow(misclassified_images(:,:,:,i));
    title(sprintf('True: %d\nPred: %d', true_labels(i), misclassified_labels(i)));
end
print -depsc E6_misclassified.eps

%% Confusion matrix
% Compute the confusion matrix
C = confusionmat(z_test, pred);

% Plot the confusion matrix with numbers
figure;
confusionchart(C);
title('Confusion Matrix');
print -depsc E6_confusion_matrix.eps

% Compute the precision and recall for each digit
precision = diag(C)./sum(C, 1)';
recall = diag(C)./sum(C, 2);

