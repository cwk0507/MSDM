% Newton's method
syms x1
syms x2
f1 = 1+x1^2/4-x2^2+exp(x1/2)*cos(x2);
f2 = x1*x2+exp(x1/2)*sin(x2);
F = [f1;f2];
J = [diff(f1,x1),diff(f1,x2);diff(f2,x1),diff(f2,x2)];
n=5;
xn(1,1:2)=[-2 4]
for i=1:n
    Jxninv = inv(vpa(subs(J,[x1,x2],xn(i,1:2))));
    Fxn = vpa(subs(F,[x1,x2],xn(i,1:2)));
    xn(i+1,1:2)=xn(i,1:2)'-Jxninv*Fxn
end;