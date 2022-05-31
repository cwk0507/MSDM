% Secant method
syms x
f = exp(x+1)/2+2^(-x)/4+cos(x+1)-3
n=4;
xn=zeros(n+2,1)
xn(1)=0.5
xn(2)=1
for i=3:n+2
    xn(i)=xn(i-1)-vpa(subs(f,x,xn(i-1)))*(xn(i-1)-xn(i-2))/(vpa(subs(f,x,xn(i-1)))-vpa(subs(f,x,xn(i-2))))
end;