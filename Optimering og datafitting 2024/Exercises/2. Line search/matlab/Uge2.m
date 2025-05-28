%%%%%%%%%%% Uge 2

%%%%%% Exercise 2.3.1
x0 = 0.1;
mu = 1.0;
alpha = 0.1;
[xopt, stat] = steepestdescent(alpha, @PenFun1, x0, mu);

k = 0:stat.iter;
err = abs(stat.X-1);

Table = [k', stat.X', err', abs(stat.dF)', stat.F'];

figure,
subplot(1,3,1), semilogy(err), title('|x_k-x^*|'),
subplot(1,3,2), plot(abs(stat.dF)), title('f''(x_k)'),
subplot(1,3,3), plot(stat.F), title('f(x_k)'),

%%%%%% Newton

x0 = 0.1;    
mu = 1.0;
alpha = 1;   
[xopt, stat] = newton(alpha, @PenFun1, x0, mu);

k = 0:stat.iter;
err = abs(stat.X-1);

Table = [k', stat.X', err', abs(stat.dF)', stat.F'];

figure,
subplot(1,3,1), semilogy(err), title('|x_k-x^*|'),
subplot(1,3,2), plot(stat.dF), title('f''(x_k)'),
subplot(1,3,3), plot(stat.F), title('f(x_k)'),


%% 2.4 
x0 = [1,0]';
alpha = 0.05;
mu = 1.0;

[xopt, stat] = steepestdescent(alpha, @MvFun, x0);

k = 0:stat.iter;
err = sqrt(stat.X(1,:).^2+stat.X(2,:).^2);
norm_df = sqrt(stat.dF(1,:).^2+stat.dF(2,:).^2);

figure,
subplot(1,3,1), plot(err), title('||x_k-x^*||'),
subplot(1,3,2), plot(norm_df), title('||f''(x_k)||'),
subplot(1,3,3), semilogy(stat.F), title('f(x_k)'),


%%%% Newton

x0 = [1,0]';
alpha = 0.05;
mu = 1.0;

[xopt, stat] = newton(alpha, @MvFun, x0);

k = 0:stat.iter;
err = sqrt(stat.X(1,:).^2+stat.X(2,:).^2);
norm_df = sqrt(stat.dF(1,:).^2+stat.dF(2,:).^2);

figure,
subplot(1,3,1), plot(err), title('||x_k-x^*||'),
subplot(1,3,2), plot(norm_df), title('||f''(x_k)||'),
subplot(1,3,3), semilogy(stat.F), title('f(x_k)'),


%% 2.5
x0 = [10;1];    
[xopt, stat] = steepestdescent_line(@MvFun, x0);

k = 0:stat.iter;
err = sqrt(stat.X(1,:).^2+stat.X(2,:).^2);
norm_df = sqrt(stat.dF(1,:).^2+stat.dF(2,:).^2);

figure,
subplot(1,3,1), plot(err), title('||x_k-x^*||'),
subplot(1,3,2), plot(norm_df), title('||f''(x_k)||'),
subplot(1,3,3), semilogy(stat.F), title('f(x_k)'),


%%% Newton

x0 = [10;1];    
[xopt, stat] = newton_line(@MvFun, x0);

k = 0:stat.iter;
err = sqrt(stat.X(1,:).^2+stat.X(2,:).^2);
norm_df = sqrt(stat.dF(1,:).^2+stat.dF(2,:).^2);

figure,
subplot(1,3,1), plot(err), title('||x_k-x^*||'),
subplot(1,3,2), plot(norm_df), title('||f''(x_k)||'),
subplot(1,3,3), semilogy(stat.F), title('f(x_k)'),


disp('Since it is a quadratic function, Newton''s method only need 2 iterations.')
disp('For this example, there are no difference for Newton''s method with or without line search.')
disp('But we need note that with line search Newton''s method can have global convergence.')


