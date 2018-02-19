function y = hessian(x,t,Q,p,A,b)
  vector = A*x-b;
  y = t*Q;
  for i = 1:size(A,1)
     y = y+(A(i,:)'*A(i,:))./(vector(i)^2);
  end
end

