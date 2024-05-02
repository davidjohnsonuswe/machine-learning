function dldX = relu_backward(X, dldY)
    dldX = dldY.*(X>0);
end
