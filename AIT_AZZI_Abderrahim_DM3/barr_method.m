function [x_sol,x_hist,gap,loss] = barr_method(Q,p,A,b,x0,mu,tol)

if (A*x0 < b) == 1
    fprintf('x0 is a feasible point\n')
end
f = @( x,t ) phi (x , t ,Q, p ,A, b) ;
g = @( x,t ) grad (x , t ,Q, p ,A, b) ; 
h = @( x,t) hessian (x , t ,Q, p ,A, b) ;
%t =0.001; 
t = 1; % Set the barrier parameter
Nmax = 10000;
m = size(A,1);
x = x0;
x_hist = x;
loss=0.5*x0'*Q*x0+p'*x0;
gap=[];
for i=1:Nmax
    fprintf('iteration %d\n',i);
    %x = newtonLS(x,t,f,g,h,A,b,tol);
    [x,~,~] = dampedNewton(x,t,f,g,h,tol);
    loss = [loss 0.5*x'*Q*x+p'*x];
    x_hist = [x_hist x];
    gap=[gap m/t];
    if  (m/t) < tol
      fprintf('Optimum found\n');
      break;
    else
        t = mu*t;
    end
end
x_sol = x;