clear all;
J=20;
delta_x=1/J;
delta_t=[0.0012,0.0013,0.012];
k=[0,1,25,50];
theta=1/2;
Nt=max(k)+1;
epsilon=1e-3;
for i=1:length(delta_t)
    mew=delta_t(i)/delta_x^2;
    d1_left(1:J)=-theta*mew;
    d2_left(1:J+1)=1+2*theta*mew;
    d3_left(1:J)=-theta*mew;
    m_left=diag(d1_left,-1)+diag(d2_left,0)+diag(d3_left,1);
    d1_right(1:J)=(1-theta)*mew;
    d2_right(1:J+1)=1-2*(1-theta)*mew;
    d3_right(1:J)=(1-theta)*mew;
    m_right=diag(d1_right,-1)+diag(d2_right,0)+diag(d3_right,1);    
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
        U(i,n+1,:)=inv(m_left)*m_right*squeeze(U(i,n,:));
        U(i,n+1,1)=0;
        U(i,n+1,J+1)=0;
        if max(abs(U(i,n+1,:)-U(i,n,:)))<epsilon
            stop=true;
        end;
    end;
end;
for mm=1:3
    for nn=1:length(k)
        subplot(length(k),3,mm+(nn-1)*3);
        plot((0:20)*delta_x,squeeze(U(mm,k(nn)+1,:)));
        title('t='+string(k(nn)*delta_t(mm))+' delta\_t='+string(delta_t(mm)));
        xlabel('x');
        ylabel('u(x,'+string(k(nn)*delta_t(mm))+')');
    end;
end;

        
    
    