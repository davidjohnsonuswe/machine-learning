%% load data
close all
load 'A2_data.mat'
X_train = train_data_01;
train_labels = train_labels_01;
X_test = test_data_01;
test_labels = test_labels_01;

%% Task E4: Assign label
K = 2;
[nbr_zeros_train,nbr_ones_train,nbr_zeros_test, ...
    nbr_ones_test,misclassified_train,misclassified_test, ...
    misclassification_rate_train,misclassification_rate_test] = ...
    evaluate_data(X_train,X_test,train_labels,test_labels,K);

%% Task E5: Try K = 1 to 10
K_array = [3,5,7,9,11,13];
misclassification_rates_train = zeros(1,length(K_array));
misclassification_rates_test = zeros(1,length(K_array));
for i=1:length(K_array)
    [~,~,~,~,~,~, ...
    misclassification_rates_train(i),misclassification_rates_test(i)] = ...
    evaluate_data(X_train,X_test,train_labels,test_labels,K_array(i));
end

%% Helper function

function [nbr_zeros_train,nbr_ones_train,nbr_zeros_test, ...
    nbr_ones_test,misclassified_train,misclassified_test, ...
    misclassification_rate_train,misclassification_rate_test] ...
= evaluate_data(X_train,X_test,train_labels,test_labels,K)

    [y_train, ~] = K_means_clustering(X_train,K);
    centroid_labels_train = K_means_classifier(train_labels,y_train,K);
    
    [y_test, ~] = K_means_clustering(X_test,K);
    centroid_labels_test = K_means_classifier(test_labels,y_test,K);
    
    % Variable for the table
    nbr_zeros_train = zeros(K,1);
    nbr_ones_train = zeros(K,1);
    nbr_zeros_test = zeros(K,1);
    nbr_ones_test = zeros(K,1);
    misclassified_train = zeros(K,1);
    misclassified_test = zeros(K,1);
    Ntrain = size(X_train,2);
    Ntest = size(X_test,2);
    
    for i=1:K
        % Total number of zeros and ones in train and test data
        nbr_zeros_train(i) = sum(train_labels(y_train==i)==0);
        nbr_ones_train(i) = sum(train_labels(y_train==i)==1);
        nbr_zeros_test(i) = sum(test_labels(y_test==i)==0);
        nbr_ones_test(i) = sum(test_labels(y_test==i)==1);
        
        % Number of misclassified cases
        if centroid_labels_train(i) == 0
            misclassified_train(i) = nbr_ones_train(i);
        else
            misclassified_train(i) = nbr_zeros_train(i);
        end
    
        if centroid_labels_test(i) == 0
            misclassified_test(i) = nbr_ones_test(i);
        else
            misclassified_test(i) = nbr_zeros_test(i);
        end
    end
    
    % Calculate sum misclassified and misclassification rate
    sum_misclassified_train = sum(misclassified_train);
    misclassification_rate_train = sum_misclassified_train./Ntrain;
    
    sum_misclassified_test = sum(misclassified_test);
    misclassification_rate_test = sum_misclassified_test./Ntest;
end
