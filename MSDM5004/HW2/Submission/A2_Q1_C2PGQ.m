% (2) compostie two-point Gauss quadrature rule
syms x c1 c2 x1 x2
f = sqrt(1+(sin(x))^3);
ns = [8 16 32];
integrals = [0 0 0];
for i=1:3
    n = ns(i);
    xa=0;
    xb=1;
    h=(xb-xa)/n;
    for j=1:n
        eq1 = c1+c2-vpa(subs(int(x^0,[xa,xa+h])));
        eq2 = c1*x1+c2*x2-vpa(subs(int(x,[xa,xa+h])));
        eq3 = c1*x1^2+c2*x2^2-vpa(subs(int(x^2,[xa,xa+h])));
        eq4 = c1*x1^3+c2*x2^3-vpa(subs(int(x^3,[xa,xa+h])));
        sol = solve(eq1,eq2,eq3,eq4);
        integrals(i) = integrals(i)+sol.c1(1)*vpa(subs(f,sol.x1(1)))+sol.c2(1)*vpa(subs(f,sol.x2(1)));
        xa = xa + h;
    end;
end;
integrals
        