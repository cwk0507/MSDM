clear all;
J=20;
delta_x=1/J;
delta_t=[0.0012,0.0013];
k=[0,1,25,50];
epsilon=1e-3;
% numerical solution
for i=1:length(delta_t)
    mew=delta_t(i)/delta_x^2;
    n=0;
    stop=false;
    for j=1:J+1
        x=(j-1)*delta_x;
        if x<=1/2
            U(i,n+1,j)=2*x;
        else
            U(i,n+1,j)=2-2*x;
        end;
    end;
    while (~stop & n<100)
        n=n+1;
        for j=1:J+1
            x=(j-1)*delta_x;
            if (x==0) | (x==1)
                U(i,n+1,j)=0;
            else
                U(i,n+1,j)=U(i,n,j)+mew*(U(i,n,j+1)-2*U(i,n,j)+U(i,n,j-1));
            end;
        end;
        if max(abs(U(i,n+1,:)-U(i,n,:)))<epsilon
            stop=true;
        end;
    end;
end;
figure(1)
for mm=1:2
    for nn=1:length(k)
        subplot(length(k),2,mm+(nn-1)*2);
        plot((0:20)*delta_x,squeeze(U(mm,k(nn)+1,:)));
        title('t='+string(k(nn)*delta_t(mm))+' delta\_t='+string(delta_t(mm)));
        xlabel('x');
        ylabel('u(x,'+string(k(nn)*delta_t(mm))+')');
    end;
end;
% analytical solution
for i=1:length(delta_t)
    n=0;
    while n<100
        for ii=1:J+1
            m=1;
            UU(i,n+1,ii)=0;
            while m<=100
                am=8/(m*pi)^2*sin(m*pi/2);
                UU(i,n+1,ii)=UU(i,n+1,ii)+am*exp(-(m*pi)^2*n*delta_t(i))*sin(m*pi*(ii-1)*delta_x);
                m=m+1;
            end;
        end;
        n=n+1;
    end;
end;
figure(2)
for mm=1:2
    for nn=1:length(k)
        subplot(length(k),2,mm+(nn-1)*2);
        plot((0:20)*delta_x,squeeze(UU(mm,k(nn)+1,:)));
        title('t='+string(k(nn)*delta_t(mm))+' delta\_t='+string(delta_t(mm)));
        xlabel('x');
        ylabel('u(x,'+string(k(nn)*delta_t(mm))+')');
    end;
end;
    
        
    
    