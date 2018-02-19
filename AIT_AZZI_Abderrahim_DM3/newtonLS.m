function [xstar,xhist,Gap] = newtonLS(x0,t,f,g,h,A,b,tol)
Gap=[];
x=x0;
Nmax=1000;
xhist=x;
k=1;
while k <= Nmax
    [x,gap] = newtonLS_step(x,t, f , g , h, A,b) ;
    if  gap < tol
      fprintf('Optimum found in %d iterations\n',k);
      break;
    end
    Gap=[Gap gap];
    xhist=[xhist x];
    k=k+1;
end
xstar=x;
end
