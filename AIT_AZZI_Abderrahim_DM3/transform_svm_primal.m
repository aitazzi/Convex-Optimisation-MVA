function [Q,p,A,b] = transform_svm_primal(tau,X,y)
[n,d]=size(X);
Q=[eye(d) zeros(d,n); zeros(n,d) zeros(n,n)];
p=[zeros(d,1);(1/(n*tau))*ones(n,1)];
b=[-1*ones(n,1); zeros(n,1)];
A = [-diag(y)*X -eye(n);zeros(n,d) -eye(n)];
end

