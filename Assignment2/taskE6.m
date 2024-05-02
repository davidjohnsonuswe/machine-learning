%% load data
close all
load 'A2_data.mat'
X_train = train_data_01';
train_labels = train_labels_01;
X_test = test_data_01';
test_labels = test_labels_01;

%% Train and predict

svm = fitcsvm(X_train, train_labels);

predict_train = predict(svm, X_train);
predict_test = predict(svm, X_test);

[nbr_zeros_train_correct,nbr_zeros_train_wrong,nbr_ones_train_correct, ...
    nbr_ones_train_wrong,nbr_zeros_test_correct,nbr_ones_test_correct, ...
    nbr_ones_test_wrong,sum_misclassified_train,sum_misclassified_test,...
    misclassification_rate_train,misclassification_rate_test] ...
= evaluate_data(predict_train,predict_test,train_labels,test_labels);

%% Helper function

function [nbr_zeros_train_correct,nbr_zeros_train_wrong,nbr_ones_train_correct, ...
    nbr_ones_train_wrong,nbr_zeros_test_correct,nbr_ones_test_correct, ...
    nbr_ones_test_wrong,sum_misclassified_train,sum_misclassified_test,...
    misclassification_rate_train,misclassification_rate_test] ...
= evaluate_data(predict_train,predict_test,train_labels,test_labels)

    Ntrain = size(predict_train,1);
    Ntest = size(predict_test,1);
    
    % Total number of zeros and ones in train and test data
    nbr_zeros_train_correct = sum(train_labels(predict_train==0)==0);
    nbr_zeros_train_wrong = sum(train_labels(predict_train==0)==1);
    nbr_ones_train_correct = sum(train_labels(predict_train==1)==1);
    nbr_ones_train_wrong = sum(train_labels(predict_train==0)==1);
    nbr_zeros_test_correct = sum(test_labels(predict_test==0)==0);
    nbr_zeros_test_wrong = sum(test_labels(predict_test==0)==1);
    nbr_ones_test_correct = sum(test_labels(predict_test==1)==1);
    nbr_ones_test_wrong = sum(test_labels(predict_test==0)==1);
    
    % Calculate sum misclassified and misclassification rate
    sum_misclassified_train = nbr_zeros_train_wrong+nbr_ones_train_wrong;
    misclassification_rate_train = sum_misclassified_train./Ntrain;
    
    sum_misclassified_test = nbr_zeros_test_wrong+nbr_ones_test_wrong;
    misclassification_rate_test = sum_misclassified_test./Ntest;
end