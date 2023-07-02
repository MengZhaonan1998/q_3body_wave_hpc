function precondOp = formPrecondOp(Kop,Cop,shift)
PR = Kop.C+0.01*eye(Kop.n); 
Pr = Kop.D+0.01*eye(Kop.m);
PR(1,:) = shift * 1i * Cop.C(1,:);
PR(end,:)=shift * 1i * Cop.C(end,:);
Pr(1,:) = shift * 1i * Cop.D(1,:);
Pr(end,:)=shift * 1i * Cop.D(end,:);
precondOp = Kron2D(Kop.a1, PR, Kop.a2, Pr, zeros(size(Kop,1),1));
end

