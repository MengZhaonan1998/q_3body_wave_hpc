function [Kop,Cop,Mop]=formKCMOp(alpha, nR, nr, LR, Lr, Vopts)
%       alpha - mass ratio M/m
%       nR - number of grid points of x coordinate
%       nr - number of grid points of y coordinate
%       LR - length of x domain [-LR,LR]
%       Lr - length of y domain [-Lr,Lr]
%       Vopts - some settings of the potential V
%            Vopts.V12
%            Vopts.V13
%            Vopts.V23
%            Vopts.q
%            Vopts.pot_type

aR = 2/(1+alpha);
ar = (1+2*alpha)/(2+2*alpha);
N = (nR+1)*(nr+1);

% assemble Chebyshev differentiation matrix D
[DR,xR]= chebD(nR);
[Dr,xr]= chebD(nr);
DR = DR/LR; 
Dr = Dr/Lr;
xR = LR*xR;
xr = Lr*xr;

% evaluate the potential function V
Vp = evalpot(nR+1,nr+1,xR,xr,Vopts);

% construct K - K posesses tensor structure
KR = DR^2;  KR(1,:) = 0;  KR(end,:) = 0;
Kr = Dr^2;  Kr(1,:) = 0;  Kr(end,:) = 0;
Kop = Kron2D(-0.5*aR, KR, -0.5*ar, Kr, Vp);

% construct C - C posesses tensor structure
CR = zeros(nR+1,nR+1);  CR(1,:) = -DR(1,:);  CR(end,:) = DR(end,:);
Cr = zeros(nr+1,nr+1);  Cr(1,:) = -Dr(1,:);  Cr(end,:) = Dr(end,:);
Cop = Kron2D(-0.5i*aR, CR, -0.5i*ar, Cr, zeros(N,1));

% construct M - M is a diagonal matrix
Mop = diagOp(-0.5);

end
