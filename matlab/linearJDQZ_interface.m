opt.pot_type = 'G';
opt.v0 = 1;
opt.q = 3;
[K,C,M] = b2d1_operator(100,opt);
[A,B] = linearize(K,C,M);

nselect=10;
sigma=0.0; 
tol=1.e-9;
method='jdqz'; J=1;
options=struct('Tol',tol,'Disp',1,'Schur','no','MaxIt',300);
options.jmin=10; options.jmax=30;
options.LSolver='exact'; 
options.LS_Tol=1.e-6; 
options.LS_MaxIt=10;  
options.Precond=[];
options.TypePrecond='i';
options.Chord=0; options.TestSpace=3;
gamma=1; 
% parameter{1}=gamma; options.par=parameter
options.par=gamma;
[Xjdqz,Lambda_jdqz]=jdqz(A,B,nselect,sigma,options);


%[Xpoly,Lambda_poly] = polyeig(K,C,M); 

%plot(Lambda_poly,'o');
%hold on;
%plot(Lambda_jdqz,'o');
