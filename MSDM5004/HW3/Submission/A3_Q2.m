format long
syms x
f = exp(-x^2);
n = [5,10,50,100,500];
integrals = zeros(1,length(n));
for i = 1:length(n)
    h = 1/n(i);
    for j = 1:n(i)
        integrals(i) = integrals(i)+(vpa(subs(f,x,(j-1)*h))+vpa(subs(f,x,j*h)))*h/2;
    end;
end;
integrals
        