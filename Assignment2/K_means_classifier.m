function centroid_label = K_means_classifier(X_label,y,K)
    
    centroid_label = zeros(K,1);
    
    % Compare the total number of zeros and ones in each cluster.
    % The larger number will be the label of the centroid.
    for i=1:K
        % Total number of zeros
        nbr_zeros = sum(X_label(y==i)==0);
        % Total number of ones
        nbr_ones = sum(X_label(y==i)==1);

        if nbr_zeros > nbr_ones
            centroid_label(i) = 0;
        else
            centroid_label(i) = 1;
        end
    end 
end