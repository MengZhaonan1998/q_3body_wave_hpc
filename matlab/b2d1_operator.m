% discretization -> quadratic eigenvalue problem K, C, M ( x in (-1,1) )
function [K,C,M] = b2d1_operator(N, opt)
if strcmp(opt.pot_type, 'G')
    Vx = G_potential(N,opt.v0);
elseif strcmp(opt.pot_type, 'L')
    Vx = L_potential(N,opt.v0,opt.q);
else 
    error('Unknown potential type');
end
D = chebD(N); 

K = -0.5*D^2 + Vx;    % construct K
K(1,:)   = D(1,:);
K(end,:) =  -D(end,:);

C = zeros(N+1,N+1);   % construct C
C(1,1) = 1i;
C(end,end) = 1i;

M = -0.5*eye(N+1);    % construct M
M(1,1) = 0;
M(end,end) = 0;

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

% Gaussian potential ( x in (-1,1) )
function [Vx] = G_potential(N,v0)
x = cos(pi*(0:N)/N);
Vx = zeros(N+1,N+1);
for i=0:N
    vala = x(i+1).^2;
    Vx(i+1,i+1) = -v0 * exp(-vala);
end
end

% Lorentzian potential ( x in (-1,1) )
function [Vx] = L_potential(N,v0,q)
x = cos(pi*(0:N)/N);
Vx = zeros(N+1,N+1);
for i=0:N
    vala = x(i+1).^2;
    Vx(i+1,i+1) = -v0*1 / ((1+vala).^q);  % typically q=3
end
end



