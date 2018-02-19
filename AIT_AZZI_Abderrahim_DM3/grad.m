function z = grad(x,t,Q,p,A,b)
vector = A*x-b;
z = t*(Q*x+p)-A'*(1./vector);
end

