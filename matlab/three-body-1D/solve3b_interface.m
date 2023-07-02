% 3b opts
alpha = 20; % M/m=20
nR = 20;    % R grid points
nr = 20;    % r grid points
LR = 5;     % R cutoff
Lr = 5;     % r cutoff

% potential opts
Vopts=[];
Vopts.V12=0;
Vopts.V13=0.05;
Vopts.V23=0.05;
Vopts.q=1.0;
Vopts.pot_type='G';

% Jacobi-Davidson opts
jdopts=[];
jdopts.numeigs=10;  % number of eigenvalues desired
jdopts.mindim=20;   % minimum dimension of search space
jdopts.maxdim=40;   % maximum dimension of search space 
jdopts.maxit=50;    % maximum number of iterations
jdopts.verbose=1;   % plot convergence(1) or not(0)
jdopts.precond=1;   % preconditioning(1) or not(0)
jdopts.tol=1e-8;    % tolerance
jdopts.sigma=0;     % target value

% gmres opts
lsopts=[];
lsopts.ltol=1e-6;   % tolerance 
lsopts.maxit=30;    % maximum number of iterations

% operator formulation 
[Kop,Cop,Mop]=formKCMOp(alpha, nR, nr, LR, Lr, Vopts);
precondOp = formPrecondOp(Kop,Cop,0.1);

% solve (K+lambda*C+lambda^2*M)psi=0
[Lambda, X, norm_r_list] = qjd(Kop, Cop, Mop, jdopts, lsopts);


