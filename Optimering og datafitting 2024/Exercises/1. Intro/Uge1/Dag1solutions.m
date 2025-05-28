%%%%%%% Dag 1 solutions

%%%% Opg 1.1
f = @(x)(x-log(x));
df = @(x) (1- 1./x);
d2f = @(x) 1./(x.^2);

x=0.01:0.01:2;
y=f(x);
dy= df(x);
d2y=d2f(x);

figure, 
subplot(1,3,1), plot(x,y,'b-',1,1,'ro'), title('f(x)'),grid on,
subplot(1,3,2), plot(x,dy), title('1st-order derivative'),grid on,
subplot(1,3,3), semilogy(x,d2y), title('2nd-order derivative'),grid on,


disp('The minimizer is x=1, which can be obtained by solving f''(x)=0.')
disp('The minimizer is unique, since d2f>0, i.e., f strictly convex.')



%%%%%%%% Opgave 1.2
disp('The gradient is [8+2x1; 12-4x2], and the Hessian is [2, 0; 0, -4].')
disp('Set the gradient equals 0, we obtain only one stationary point [-4;3].')
disp('Since Hessian is neither positive definite nore negative definite, the stationary point is just a saddle point.')

x1=-9:0.05:1;
x2=-2:0.05:8;
[X,Y]=meshgrid(x1,x2);
F=8*X+12*Y+X.^2-2*Y.^2;

figure,
v=[-20:20];
[c,h]=contour(X,Y,F,v,'linewidth',2);
colorbar, axis image,
xlabel('x_1','fontsize',14),
ylabel('x_2','fontsize',14),


%%%%%%% Opgave 4.2
clear all, close all,

%% f1
f1 = @(x)(x.^2+x+1);
df1 = @(x) (2*x+1);
d2f1 = @(x) 2*ones(length(x));

x=-2:0.01:2;
y1=f1(x);
dy1= df1(x);
d2y1=d2f1(x);

figure, 
subplot(1,3,1), plot(x,y1,'b-'), title('f(x)'),grid on,
subplot(1,3,2), plot(x,dy1), title('1st-order derivative'),grid on,
subplot(1,3,3), semilogy(x,d2y1), title('2nd-order derivative'),grid on,


%% f2

f2 = @(x)(-x.^2+x+1);
df2 = @(x) (-2*x+1);
d2f2 = @(x) -2*ones(length(x));

x=-2:0.01:2;
y2=f2(x);
dy2= df2(x);
d2y2=d2f2(x);

figure, 
subplot(1,3,1), plot(x,y2,'b-'), title('f(x)'),grid on,
subplot(1,3,2), plot(x,dy2), title('1st-order derivative'),grid on,
subplot(1,3,3), semilogy(x,d2y2), title('2nd-order derivative'),grid on,

%% f3

f3 = @(x)(x.^3-5*x.^2+x+1);
df3 = @(x) (3*x.^2-10*x+1);
d2f3 = @(x) 6*x-10;

x=-4:0.01:4;
y3=f3(x);
dy3= df3(x);
d2y3=d2f3(x);

figure, 
subplot(1,3,1), plot(x,y3,'b-'), title('f(x)'),grid on,
subplot(1,3,2), plot(x,dy3), title('1st-order derivative'),grid on,
subplot(1,3,3), semilogy(x,d2y3), title('2nd-order derivative'),grid on,


%% f4

f4 = @(x)(x.^4+x.^3-10*x.^2-x+1);
df4 = @(x) (4*x.^3+3*x.^2-20*x-1);
d2f4 = @(x) 12*x.^2+6*x-20;

x=-4:0.01:4;
y4=f4(x);
dy4= df4(x);
d2y4=d2f4(x);

figure, 
subplot(1,3,1), plot(x,y4,'b-'), title('f(x)'),grid on,
subplot(1,3,2), plot(x,dy4), title('1st-order derivative'),grid on,
subplot(1,3,3), semilogy(x,d2y4), title('2nd-order derivative'),grid on,


disp('f1 is convex.')
disp('At local minimizer, the value of the 2nd derivative is positive.')
dips('At local maximizer, the value of the 2nd derivative is negative.')


%%%%%% Opgave 4.5
x1=0:0.05:5;
x2=x1;
[X,Y]=meshgrid(x1,x2);
F=X.^2+Y.^2;

figure,
v=[-20:20];
[c,h]=contour(X,Y,F,v,'linewidth',2);
colorbar, axis image,
xlabel('x_1','fontsize',14),
ylabel('x_2','fontsize',14),

disp('The feasible set is a convext set.')

%%

f = @(x)(x.*log(x));
df = @(x)(log(x)+1);
d2f = @(x)(1./x);

x=0.01:0.01:5;
y=f(x);
dy= df(x);
d2y=d2f(x);

figure, 
subplot(1,3,1), plot(x,y,'b-',1,1), title('f(x)'),grid on,
subplot(1,3,2), plot(x,dy), title('1st-order derivative'),grid on,
subplot(1,3,3), semilogy(x,d2y), title('2nd-order derivative'),grid on,

disp('The objective function is a convex function in the feasible set.')