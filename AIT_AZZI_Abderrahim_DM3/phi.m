function z = phi(x,t,Q,p,A,b)
z=t*(0.5*x'*Q*x + p'*x)-sum(log(b-A*x));
end

