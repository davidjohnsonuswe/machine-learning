%% Compute the kernel matrix K using the data from the table

x = [-2,-1,1,2];
k = zeros(4);
for i = 1:4
    for j = 1:4
        k(i,j) = x(i)*x(j) + (x(i)*x(j))^2;
    end
end