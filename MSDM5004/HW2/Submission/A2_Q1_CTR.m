% (1) compostie trapezoidal rule
syms x
f = sqrt(1+(sin(x))^3);
ns = [8 16 32];
integrals = [0 0 0];
for i = 1:3
    n = ns(i);
    x0 = 0;
    x1 = 1;
    h=(x1-x0)/n;
    for j=1:n
        integrals(i)=integrals(i)+vpa(subs(f,x0))+vpa(subs(f,x0+h));
        x0=x0+h;
    end;
    integrals(i)=integrals(i)*h/2;
end;
integrals


