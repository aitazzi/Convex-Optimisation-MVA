function [Q,p,A,b] = transform_svm_dual(tau,X,y)
[n,d]=size(X);
Q=diag(y)*X*(X')*diag(y);
p=-1*ones(n,1);
b=[(1/(n*tau))*ones(n,1); zeros(n,1)];
A = [eye(n);-1*eye(n)];
end

