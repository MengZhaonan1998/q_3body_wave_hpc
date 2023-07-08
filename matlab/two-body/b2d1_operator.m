function [K,C,M] = b2d1_operator(L, N, opt)
% discretization of two-body resonance system 
% output:
% quadratic eigenvalue problem K, C, M ( x in (-L,L) ) 
% input: 
% L->length of domain [-L,L]
% N->number of grid points
% opt-> other options such as potential type etc.

% chebyshev differentiation matrix
[D,x] = chebD(N);          
D = D/L;
x = L*x;

if strcmp(opt.pot_type, 'G')
    Vx = G_potential(x,N,opt.v0);       % Gaussian potential
elseif strcmp(opt.pot_type, 'L') 
    Vx = L_potential(x,N,opt.v0,opt.q); % Lorentzian potential
elseif strcmp(opt.pot_type, 'square')
    Vx = S_potential(x,N,opt.v0);
elseif strcmp(opt.pot_type, 'None')
    Vx = zeros(N+1);
else 
    error('Unknown potential type');
end

% construct K
K = -0.5*D^2 + Vx;    
K(1,:)   =  D(1,:);
K(end,:) = -D(end,:);

% construct C
C = zeros(N+1,N+1);   
C(1,1) = 1i;
C(end,end) = 1i;

% construct M
M = -0.5*eye(N+1);    
M(1,1) = 0;
M(end,end) = 0;

end

% Gaussian potential
function [Vx] = G_potential(x,N,v0)
Vx = zeros(N+1);
for i=0:N
    vala = x(i+1).^2;
    Vx(i+1,i+1) = -v0 * exp(-vala);
end
end

% Lorentzian potential 
function [Vx] = L_potential(x,N,v0,q)
Vx = zeros(N+1);
for i=0:N
    vala = x(i+1).^2;
    Vx(i+1,i+1) = -v0*1 / ((1+vala).^q);  % typically q=3
end
end

% Square potential
function [Vx] = S_potential(x,N,v0)
Vx = zeros(N+1);
for i=0:N
    if x(i+1)>=-1 && x(i+1)<=1
        Vx(i+1,i+1) = v0;
    end
end
end

% Chebyshev differentiation matrix ( x in (-1,1) )
function [D,x] = chebD(N)
x = cos(pi*(0:N)/N)'; 
c = [2; ones(N-1,1); 2].*(-1).^(0:N)';
X = repmat(x,1,N+1);
dX = X-X';                  
D  = (c*(1./c)')./(dX+(eye(N+1)));       % off-diagonal entries
D  = D - diag(sum(D,2));                 % diagonal entries
x = x(N+1:-1:1);
D = D(N+1:-1:1,N+1:-1:1);
end

