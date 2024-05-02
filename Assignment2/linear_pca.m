function proj_data = linear_pca(X)
    % Normalizie X so that the mean of each row is 0
    X_norm = X - mean(X,2);
    % Calculate the left singular vectors
    [U,~,~] = svd(X_norm);
    % Calculate the projection onto the first and second principal component
    U_d = U(:,1:2);
    proj_data = U_d'*X_norm;
end