% (2) SOR method
A = [0 -1/4 1/4 -1/4 -3/4; -1/4 0 1/4 1/4 -1/2; 1/5 1/5 0 -1/5 1; -1/3 1/3 -1/3 0 2/3];
x = [0 0 0 0 1]';
x_sol = [0 0 0 0]';
w=1.2;
epsilon = 1e-3;
for i=1:4  
    x(i) = (1-w)*x(i)+w*dot(A(i,:),x);
end;
x_sol(:,2) = x(1:4);
c = 2;
while norm(x_sol(:,c)-x_sol(:,c-1),Inf)>epsilon
    for i=1:4
        x(i) = (1-w)*x(i)+w*dot(A(i,:),x);
    end;
    c=c+1;
    x_sol(:,c)=x(1:4);
end;
x_sol(1:4,:)