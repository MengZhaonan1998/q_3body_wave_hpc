function [Lambda,X,norm_r_list] = jdpoly(P,K,SIGMA,opts)
% nonlinear Jacobi Davidson algorithm
% JDpoly computes several eigenpairs of quadratic eigenvalue problem (lambda^2*A2*x+lambda*A1*x+A0*x=0)
% [Lambda,X]=jdpoly(P,K,SIGMA,opts)

% input:
%      P(struct)- the struct of coefficient matrices of the polynomial eigenvalue problem  
%          P.A0 - stiffness matrix K
%          P.A1 - damping matrix C
%          P.A2 - mass matrix M
%      K         an integer, the number of desired eigenvalues.
%      SIGMA     a scalar shift or a two letter string.
%      opts      a structure containing additional parameters.

% output:
%    Lambda - eigenvalue
%    X - eigenvector 
%    hist - convergence history

% some options of the algorithm
% opts.mindim [10]  min dimension of search space
% opts.maxdim [25]  max dimension of search space
% opts.tol  [1e-8]  residual tolerance
% opts.maxit [100]  maximum number of iterations
% opts.v0  ones+0.1*rand          Starting space                        

A0 = P.A0;  A1 = P.A1;  A2 = P.A2;  % A0,A1,A2 matrix stored in struct P
N = size(A0,1);                     % problem size
if nargin < 2
    K = 5;
    SIGMA = 0+0i;
end
if nargin < 3
    % some settings    
    mindim = 10; maxdim = 25;
    tol = 1.0e-8; maxit = 100;
    v0 = 0.5*ones(N,1) + 0.1*rand(N,1);
    lsolver = 'gmres';
    precond = [];
    verbose = 1;
else
    mindim = getopt(opts,'mindim',10);
    maxdim = getopt(opts,'maxdim',25);
    tol    = getopt(opts,'tol',1.0e-8);
    maxit  = getopt(opts,'maxit',100);
    v0     = getopt(opts,'v0',ones(N,1));
    lsolver= getopt(opts,'lsolver','gmres');
    precond= getopt(opts,'precond',[]);
    verbose= getopt(opts,'verbose',1);
end

t = v0; V = [];      
Lambda = []; X = [];            % convergence history
iter = 0; detected = 0;         % number of eigenpairs detected 
norm_r_list = [];

t = t/norm(t);                  % normalize t
V = [V,t];                      % search space V = t

while (detected < K && iter < maxit )

    %%% standard rayleigh ritz projection
    H0 = V' * A0 * V;
    H1 = V' * A1 * V;
    H2 = V' * A2 * V;

    %%% solve the projected problem
    %[theta,c] = linear_eig(H0,H1,H2);  
    % Using linearization to solve the smaller problem
    [c,theta] = polyeig(H0,H1,H2);
    v = V * c;                                  % v = V_k * c
    
    %%% Sort the eigenpairs in terms of distance with target tau
    [theta,v] = eig_sortsigma(theta,v,SIGMA);   %[theta,v] = eig_sortdescent(theta,v);
    theta_best = theta(detected + 1);             % best Ritz pair (theta,v)
    v_best = v(:,detected + 1);
   
    %%% residual
    r = (A0 + theta_best * A1 + theta_best * theta_best * A2) * v_best;  
    norm_r = norm(r);
    norm_r_list = [norm_r_list,norm_r];

    %%% stopping criteria
    if norm_r < tol                
        Lambda = [Lambda, theta_best]; 
        X = [X, v_best];
        detected = detected + 1;
        V(:,detected) = v_best; % add converged schur to the search space
        tau = theta_best;

        %%% reduce the search space if necessary
        if size(V,2) >= maxdim
            dim = min(mindim, size(v,2));
            V = v(:,1:dim);
            V(:,1:length(Lambda)) = X;  % add converged eigenvectors in the search space V
        end

        theta_best = theta(detected+1);
        v_best = v(:,detected+1);
        r = (A0 + theta_best * A1 + theta_best * theta_best * A2) * v_best; 
    end

    %%% restart when dimension of V is too large
    %if size(V,2) == maxdim
    %    dim = min(mindim, size(v,2));    % in case that number of ritz vectors is less than minimum dimension
    %    V = v(:,1:dim);                  % new search space V 
    %    if ~isempty(Lambda)
    %        V(:,1:length(Lambda)) = X;
    %    end
    %end
  
    %%% Solve the (preconditioned) correction equation
    %t = gmres(corrA,-r,[],1e-6,100);   
    z = (2 * theta_best * A2 + A1) * v_best;

    [t, xtol] = SolvePCE(theta_best, v_best, z, r, A0, A1, A2, lsolver,precond,SIGMA);
    
    %%% Expand the subspaces and the interaction matrices
    V = [V,t]; 
    % we can use either repeat gs or modified gs ...
    %V = modified_gs(V,size(V,2));
    V = RepGS(V);  
    
    %%% determine a new preconditioner if necessary
    %K_pre = theta_best^2 * A2 + theta_best * A1 + A0;
    %[L,U]=lu(theta_best^2 * A2 + theta_best * A1 + A0);
    %K_pre.L = L;
    %K_pre.U = U;

    %%% display current status
    fprintf('iteration:%d, residual norm:%.3e, lsolver rnorm:%.3e \n',iter,norm_r,xtol);
    
    iter = iter + 1;  % number of iteration +1
end

if verbose==1
    iter_step = (0:size(norm_r_list,2)-1);
    p=semilogy(iter_step,norm_r_list,'*-',iter_step,ones(size(norm_r_list,2))*opts.tol,'r:');
    p(1).LineWidth = 1; 
    p(2).LineWidth = 2;
    xlabel('number of iterations');
    ylabel('log_{10} || r ||_2');
    title('Quadratic Jacobi-Davidson');
    grid on;
end

return
%%%================= Quadratic Jacobi-Davidson end ========================


%%%============== (preconditioned) correction equation solver==============
function [t,xtol]=SolvePCE(theta,u,w,r,A0,A1,A2,lsover,precond,SIGMA)
N = length(u);
op1 = eye(N) - (w * u')/(u' * w);
op2 = A0 + theta*A1;
op3 = eye(N) - (u * u')/(u' * u);

A = op1 * op2 * op3;   %cond(A) -> condition number
b = -r;

switch precond
    case {'shiftH'}
        K_pre.L = A0 + theta*A1 + theta^2*A2;
        K_pre.U = eye(N);
    case {'M'}
        K_pre.L = SIGMA^2 * A2 + SIGMA * A1 + A0;
        K_pre.U = eye(N);
    case {'lu'}
        [L,U] = lu(SIGMA^2 * A2 + SIGMA * A1 + A0);
        K_pre.L = L;
        K_pre.U = U;
    case {'jacobi'}
        K_pre.L = diag(diag(A));
        K_pre.U = eye(N);
    case {'gs'}
        K_pre.L = tril(A);
        K_pre.U = eye(N);
    case {' '}
        K_pre.L = eye(N);
        K_pre.U = eye(N);
end

maxit = 30;
tol = 1e-6;
x0 = zeros(N,1);

switch lsover
    case {'exact'}   % only for experimental purpose
        t = A\b;
        xtol = norm(A*t-b);
    case {'gmres'}
        gmres_restart = [];
        t = gmres(A,b,gmres_restart,tol,maxit,K_pre.L,K_pre.U);
        xtol = norm(A*t-b);
    case {'gmres0'}
        [t, e] = gmres0( K_pre.U^-1*K_pre.L^-1*A, K_pre.U^-1*K_pre.L^-1*b, x0, maxit, tol);
        xtol = e(end);
    case {'bicgstab'}
        t = bicgstab(A,b,tol,maxit,K_pre.L,K_pre.U);
        xtol = norm(A*t-b);
    case {'bicgstab0'}
        [t, xtol, iter, flag] = bicgstab0(A, x0, b, K_pre.L*K_pre.U, maxit, tol);
end

return
%%%========================================================================

%%%===========================Linearization================================
function [theta,c]=linear_eig(K,C,M)
% linear_eig(H0,H1,H2,neigs) uses built-in eig to compute the eigen pairs
% of the projected QEP via a generalized linear eigenvalue probelm.
% M,C,K: coefficients of the given QEP l^2*M+l*C+K=0
% (theta,c) is the Ritz pair

N = length(K);
I = speye(N);
Z = spalloc(N,N,0);

if issparse(K) == 0
    % non-sparse matrix
    Z = zeros(N);
    I = eye(N);
    A = [C,K;-I,Z];
    B = [-M,Z;Z,-I];
    [c,theta]=eig(A,B);
else
    % sparse matrix
    Z = spalloc(N,N,0);
    I = speye(N);
    A = [C,K;-I,Z];
    B = [-M,Z;Z,-I];
    [c,theta]=eigs(A,B);
end

theta = diag(theta);
c = c(1:N,:);      % extract eigenvector c from [theta*c;c]
c = c./vecnorm(c); % normalize each column of the matrix c

return

% sort eigenvalue based on distance with sigma
function [theta,c]=eig_sortsigma(theta,c,tau)
[distance,i]=sort(abs(theta-tau));
theta = theta(i); 
c = c(:,i);
return
% sort eigenvalue based on absolute value
function [theta,c]=eig_sortdescent(theta,c)
[theta,i]=sort(abs(theta)); 
c = c(:,i);
return

%%%========================================================================   

%%%==========================get opts struct ==============================
function value=getopt(opts,name,default)
value=default;
if isfield(opts,name)
    value=getfield(opts,name);
end
return
%%%========================================================================

%%%========================= Orthogonalisation ============================
function [V,R]=RepGS(Z,V,gamma)
%
% Orthonormalisation using repeated Gram-Schmidt
%   with the Daniel-Gragg-Kaufman-Stewart (DGKS) criterion
%
% Q=RepGS(V)
%   The n by k matrix V is orthonormalized, that is,
%   Q is an n by k orthonormal matrix and 
%   the columns of Q span the same space as the columns of V
%   (in fact the first j columns of Q span the same space
%   as the first j columns of V for all j <= k).
%
% Q=RepGS(Z,V)
%  Assuming Z is n by l orthonormal, V is orthonormalized against Z:
%  [Z,Q]=RepGS([Z,V])
%
% Q=RepGS(Z,V,gamma)
%  With gamma=0, V is only orthogonalized against Z
%  Default gamma=1 (the same as Q=RepGS(Z,V))
%
% [Q,R]=RepGS(Z,V,gamma)
%  if gamma == 1, V=[Z,Q]*R; else, V=Z*R+Q; end
 
% coded by Gerard Sleijpen, March, 2002

if nargin == 1, V=Z; Z=zeros(size(V,1),0); end   
if nargin <3, gamma=1; end

[n,dv]=size(V); [m,dz]=size(Z);

if gamma, l0=min(dv+dz,n); else, l0=dz; end
R=zeros(l0,dv); 

if dv==0, return, end
if dz==0 & gamma==0, return, end

% if m~=n
%   if m<n, Z=[Z;zeros(n-m,dz)]; end
%   if m>n, V=[V;zeros(m-n,dv)]; n=m; end
% end

if (dz==0 & gamma)
   j=1; l=1; J=1;
   q=V(:,1); nr=norm(q); R(1,1)=nr;
   while nr==0, q=rand(n,1); nr=norm(q); end, V(:,1)=q/nr;
   if dv==1, return, end    
else
   j=0; l=0; J=[];
end

while j<dv,
   j=j+1; q=V(:,j); nr_o=norm(q); nr=eps*nr_o;
   if dz>0, yz=Z'*q;     q=q-Z*yz;      end
   if l>0,  y=V(:,J)'*q; q=q-V(:,J)*y;  end
   nr_n=norm(q);
  
   while (nr_n<0.5*nr_o & nr_n > nr)
      if dz>0, sz=Z'*q;     q=q-Z*sz;     yz=yz+sz; end
      if l>0,  s=V(:,J)'*q; q=q-V(:,J)*s; y=y+s;    end
      nr_o=nr_n; nr_n=norm(q);                
   end
   if dz>0, R(1:dz,j)=yz; end
   if l>0,  R(dz+J,j)=y;  end

   if ~gamma
     V(:,j)=q;
   elseif l+dz<n, l=l+1; 
     if nr_n <= nr % expand with a random vector
       % if nr_n==0
       V(:,l)=RepGS([Z,V(:,J)],rand(n,1));
       % else % which can be numerical noice
       %   V(:,l)=q/nr_n;
       % end
     else
       V(:,l)=q/nr_n; R(dz+l,j)=nr_n; 
     end
     J=[1:l];
   end 

end % while j

if gamma & l<dv, V=V(:,J); end

return
%%%=====================END  Orthogonalisation=============================

%%%======================modified gram schmidt=============================
function Q=modified_gs(X,n)
for j=1:n
    V(:,j)=X(:,j);
end
for j=1:n
    Q(:,j)=V(:,j)/norm(V(:,j));
    for k=j+1:n
        V(:,k)=V(:,k)-Q(:,j)'*V(:,k)*Q(:,j);
    end
end
return