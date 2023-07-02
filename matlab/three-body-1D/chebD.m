% Chebyshev differentiation matrix ( x in (-1,1) )
function [D,x] = chebD(N)
x = cos(pi*(0:N)/N)'; 
c = [2; ones(N-1,1); 2].*(-1).^(0:N)';
X = repmat(x,1,N+1);
dX = X-X';                  
D  = (c*(1./c)')./(dX+(eye(N+1)));    % off-diagonal entries
D  = D - diag(sum(D,2));              % diagonal entries
x = x(N+1:-1:1);                      % order reverse
D = D(N+1:-1:1,N+1:-1:1);             % order reverse 
return