d1(1:6)=1;
d2=[1,1,0,1,1,0,1,1];
d3(1:9)=-4;
A=diag(d1,-3)+diag(d1,3)+diag(d2,-1)+diag(d2,1)+diag(d3,0);
D=diag(d3,0);
b=[-0.75,-0.875,-0.75,-0.875,-1,-0.875,-0.75,-0.875,-0.75]';
x0=zeros(9,1);
epsilon=1e-5;
stop=false;
n=0;
while ~stop
    n=n+1;
    x1=inv(D)*(A-D)*x0+inv(D)*b;
    if max(abs(x1-x0))<epsilon
        stop=true;
    end;
    x0=x1;
end;
x0
n  