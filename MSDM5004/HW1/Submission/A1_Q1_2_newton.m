% Newton's method
syms x
f = exp(x+1)/2+2^(-x)/4+cos(x+1)-3
f1 = diff(f)
n=4;
xn=zeros(n+1,1)
xn(1)=0.5
for i=2:n+1
    xn(i)=xn(i-1)-vpa(subs(f,x,xn(i-1)))/vpa(subs(f1,x,xn(i-1)))
end;