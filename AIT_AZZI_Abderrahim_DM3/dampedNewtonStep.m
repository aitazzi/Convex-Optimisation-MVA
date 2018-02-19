function [xnew,gap] = dampedNewtonStep(x,t,f,g,h)
% Compute the newton step and decrement
v=-h(x,t)\g(x,t);
lambda2=-g(x,t)'*v;
s=1/(1+sqrt(lambda2));
xnew=x+s*v; 
gap=lambda2/2;
end

