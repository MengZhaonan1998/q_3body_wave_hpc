n = 10;
A = 10*rand(n);
b = 10*rand(n,1);
x = ones(n,1)+0.1*rand(n,1);
M = eye(n);
restrt = 20;
max_it = 100;
tol = 1e-6;

[x, e] = gmres( A,b,x,max_it,tol );
