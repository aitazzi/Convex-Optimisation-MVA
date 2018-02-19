function [xnew,gap] = newtonLS_step(x,t,f,g,h,A,b)
% Compute the newton step and decrement
v=-h(x,t)\g(x,t);
lambda2=-g(x,t)'*v;
alpha=0.01;
beta=0.05;

s=1;
while 1
    if (f(x+s*v,t)> f(x,t) - alpha*s*lambda2) || sum(A*(x+s*v) > b)>0 % The second condition to stay in the domain
        s = beta*s;
    else
        break;
    end
end
xnew=x+s*v;
gap=lambda2/2;
end


