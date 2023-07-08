opt.pot_type = 'G';
opt.v0 = 0.34459535;
opt.q = 3;
[K,C,M] = b2d1_operator(3,2048,opt);
P.A0 = K;
P.A1 = C;
P.A2 = M;
N = size(K,1);  

sigma = -1.0899-1.6329i;
opts.mindim=10;
opts.maxdim=30;
opts.maxit=50;
opts.v0=ones(N,1);% + 0.1*rand(N,1);
opts.lsolver='gmres';
opts.verbose=1;
opts.tol=1e-8;


%opts.precond=' ';
%[Lambda,X_pre,norm_r_list_2_2048] = jdpoly(P,30,sigma,opts); % with preconditioning


tic
opts.precond='shiftH';
[Lambda,X_pre,norm_r_list_1_2048] = jdpoly(P,30,sigma,opts); % with preconditioning
toc
%iter_step_1 = (1:size(norm_r_list_1,2));
%iter_step_2 = (1:size(norm_r_list_2,2));
%p1=semilogy(iter_step_1,norm_r_list_1/50,'*-',iter_step_2,norm_r_list_2/50,'*-',iter_step_1,ones(size(norm_r_list_1,2))*opts.tol/50,'r:');
%p1(1).LineWidth = 1; 
%p1(2).LineWidth = 1;
%xlabel('number of iterations');
%ylabel('||\itr||_2');
%title('Quadratic Jacobi-Davidson');
%grid on;






%opts.v0=X_pre;
%opts.tol=1e-6;
%[Lambda,X,norm_r_list_2] = jdpoly(P,5,sigma,opts); % with preconditioning


%opts.precond='[]';
%[Lambda,X,norm_r_list_1] = jdpoly(P,1,0+0i,opts); % without preconditioning


%iter_step_1 = (0:size(norm_r_list_1,2)-1);
%iter_step_2 = (0:size(norm_r_list_2,2)-1);
%p=semilogy(iter_step_1,norm_r_list_1,'*-', ...
%           iter_step_2,norm_r_list_2,'*-', ...
%           (0:opts.maxit),ones(opts.maxit+1)*opts.tol,'r:');
%p(1).LineWidth = 1;
%p(2).LineWidth = 1;
%p(3).LineWidth = 2;
%xlabel('number of iterations');
%ylabel('log_{10} || r ||_2');
%title('Quadratic Jacobi-Davidson');
%legend('non-prec-JD','LU-prec-JD')
%grid on;

