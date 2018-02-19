function [xstar,xhist,Gap] = dampedNewton(x0,t,f,g,h,tol)
x=x0;
Nmax=1000;
Gap=[];
xhist=x;
for i=1:Nmax
    [x,gap] = dampedNewtonStep (x,t, f , g , h) ;
    Gap=[Gap gap];
    if  gap < tol
      fprintf('Optimum found in %d iterations\n',i);
      break;
    end
    xhist=[xhist x];
end
xstar=x;

