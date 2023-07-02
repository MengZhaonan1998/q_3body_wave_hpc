function [Lambda, X, norm_r_list] = qjd(Kop, Cop, Mop, jdopts, lsopts)
% Quadratic Jacobi-Davidson algorithm for computing one dimensional three body problem 
% coded by Zhaonan Meng, 2023 

% [Lambda, X, norm_r_list] = jdpoly_3b1d(nR, nr, LR, Lr, alpha1, alpha2, Vopts, jdopts, lsopts)
% input:
%       Kop - stiffness operator     
%       Cop - damping operator
%       Mop - mass operator
%       jdopts - some settings of the jacobidavidson algorithm
%            jdopts.numeigs      number of eigenvalues desired
%            jdopts.mindim [10]  min dimension of search space
%            jdopts.maxdim [30]  max dimension of search space
%            jdopts.tol  [1e-8]  residual tolerance
%            jdopts.maxit [100]  maximum number of iterations            
%            jdopts.v0 ones(N,1) starting space                       
%            jdopts.lsolver      linear solver
%            jdopts.precond      preconditioner type
%            jdopts.verbose      print convergence history or not
%            jdopts.sigma        target
%       lsopts - some settings of the linear solver GMRES
%            lsopts.ltol [1e-5]  tolerance 
%            lsopts.lmaxit [30]  maximum number of GMRES iteration 
% output:
%       Lambda - eigenvalue
%       X - eigenvector 
%       norm_r_list - convergence history

N = size(Kop,1);  % total number of grid points 
% get settings of JD alg
numeigs= getopt(jdopts,'numeigs',10);
mindim = getopt(jdopts,'mindim',10);
maxdim = getopt(jdopts,'maxdim',30);
tol    = getopt(jdopts,'tol',1.0e-6);
maxit  = getopt(jdopts,'maxit',100);
v0     = getopt(jdopts,'v0',ones(N,1));
precond= getopt(jdopts,'precond',0);
verbose= getopt(jdopts,'verbose',1);
sigma  = getopt(jdopts,'sigma',0+0i); 
ltol   = getopt(lsopts,'ltol',1e-5);  
lmaxit = getopt(lsopts,'lmaxit',30); 

t = v0;           % starting vector 
V = [];           % search space
Lambda = [];      % eigenvalue list
X = [];           % eigenvector list
iter = 0;         % number of iteration
detected = 0;     % number of eigenpairs detected 
norm_r_list = []; % convergence history
tau = sigma;      % target value

t = t/norm(t);    % normalize t
V = [V,t];        % search space V = t

%%% Jacobi-Davidson iteration starts
while (detected < numeigs && iter < maxit )
    
    %%% standard rayleigh ritz projection
    H0 = V' * mtimes(Kop,V);
    H1 = V' * mtimes(Cop,V);
    H2 = V' * mtimes(Mop,V);

    %%% solve the projected problem - using linearization to solve the smaller problem
    [c,theta] = polyeig(H0,H1,H2);
    %[A,B] = linearize(H0,H1,H2);
    %[c,theta] = eig(A,B);
    %theta = diag(theta);
    %c = c(end/2+1:end,:);
    v = V * c;                                  % v = V_k * c

    %%% Sort the eigenpairs in terms of distance with target tau
    [theta,v] = eig_sortsigma(theta,v,tau);     %[theta,v] = eig_sortdescent(theta,v);
    theta_best = theta(detected + 1);           % best Ritz pair (theta,v)
    v_best = v(:,detected + 1);
   
    %%% residual
    r = mtimes(Kop,v_best) + theta_best * mtimes(Cop,v_best) + theta_best * theta_best * mtimes(Mop,v_best); 
    norm_r = norm(r);
    norm_r_list = [norm_r_list,norm_r];

    %%% stopping criteria
    if norm_r < tol                
        Lambda = [Lambda, theta_best]; 
        X = [X, v_best];
        detected = detected + 1;
        V(:,detected) = v_best;   % add converged schur to the search space
        tau = theta_best;
        theta_best = theta(detected+1);
        v_best = v(:,detected+1);
        r = mtimes(Kop,v_best) + theta_best * mtimes(Cop,v_best) + theta_best * theta_best * mtimes(Mop,v_best); 
    end

    %%% restart when dimension of V is too large
    if size(V,2) == maxdim
        dim = min(mindim, size(v,2));    % in case that number of ritz vectors is less than minimum dimension
        V = v(:,1:dim);                  % new search space V 
        if ~isempty(Lambda)
            V(:,1:length(Lambda)) = X;
        end
    end
    
    %%% Solve the (preconditioned) correction equation   
    z = 2 * theta_best * mtimes(Mop,v_best) + mtimes(Cop,v_best);
    if precond == 0
        % solve correction equation without preconditioning
        correction_op = correctOp(Kop, Cop, Mop, v_best, z, theta_best);
        t = zeros(N,1);
        [t, xtol] = gmres0(correction_op, t, -r, ltol, lmaxit);
        xtol=xtol(end);
        fprintf('iteration:%d, residual norm:%.3e, correction rnorm:%.3e \n',iter,norm_r,xtol);
    elseif precond ==1
        % solve correction equation with preconditioning (I-wu/uw)K(I-uu/uu)
        Pop = formPrecondOp(Kop,Cop,theta_best);
        z_tilde = mldivide(Pop,z);
        mu = v_best' * z_tilde;
        r_tilde = mldivide(Pop,r);
        r_tilde = r_tilde - v_best' * r_tilde * z_tilde / mu;
        t = zeros(N,1);
        correction_op = preCorrOp(Kop, Cop, Mop, theta_best, Pop, v_best, z, z_tilde, mu);
        [t, xtol] = gmres0(correction_op, t, -r_tilde, ltol, lmaxit);
        pxtol=xtol(end);
        fprintf('iteration:%d, residual norm:%.3e, (preconditioned)correction rnorm:%.3e \n',iter,norm_r,pxtol);
    end
    
    %%% Expand the subspaces and the interaction matrices
    V = [V,t]; 
    %V = modified_gs(V,size(V,2)); % we can use either repeat gs or modified gs ...
    V = RepGS(V);  
    
    iter = iter + 1;  % number of iteration +1
end

if verbose==1
    iter_step = (0:size(norm_r_list,2)-1);
    p=semilogy(iter_step,norm_r_list,'*-',iter_step,ones(size(norm_r_list,2))*jdopts.tol,'r:');
    p(1).LineWidth = 1; 
    p(2).LineWidth = 2;
    xlabel('number of iterations');
    ylabel('log_{10} || r ||_2');
    title('Quadratic Jacobi-Davidson');
    grid on;
end

return


%%% sort eigenvalue based on distance with sigma
function [theta,c]=eig_sortsigma(theta,c,tau)
[distance,i]=sort(abs(theta-tau));
theta = theta(i); 
c = c(:,i);
return

%%% sort eigenvalue based on absolute value
function [theta,c]=eig_sortdescent(theta,c)
[theta,i]=sort(abs(theta)); 
c = c(:,i);
return  

%%% get opts struct 
function value=getopt(opts,name,default)
value=default;
if isfield(opts,name)
    value=getfield(opts,name);
end
return

