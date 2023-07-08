opt.pot_type = 'G';
opt.v0 = 0.34459535;
opt.q = 3;
n=128;
[K,C,M] = b2d1_operator(5,n,opt);
[A,B] = linearize(K,C,M);

nselect=10;
sigma=0-1i; 
tol=1.e-8;
method='jdqz'; J=1;

options=struct('Tol',tol,'Disp',1,'Schur','no','MaxIt',50);
options.jmin=10; options.jmax=30;
options.LSolver='gmres'; 
options.LS_Tol=1.e-6; 
options.LS_MaxIt=30;  
options.Chord=0; options.TestSpace=3;
gamma=1; 
options.par=gamma;
options.Precond=A-sigma*B;
options.v0=ones(size(A,1),1);
options.TypePrecond='left';
tic
[X,JORDAN,Q,Z,S,T,HISTORY_2_128]=jdqz(A,B,nselect,sigma,options);
toc
%options2=struct('Tol',tol,'Disp',1,'Schur','no','MaxIt',50);
%options2.jmin=10; options2.jmax=30;
%options2.LSolver='gmres'; 
%options2.LS_Tol=1.e-6; 
%options2.LS_MaxIt=30;  
%options2.Chord=0; options2.TestSpace=3;
%gamma=1; 
%options2.par=gamma;
%options2.TypePrecond=[];
%tic
%[X,JORDAN,Q,Z,S,T,HISTORY_1_128]=jdqz(A,B,nselect,sigma,options2);
%toc

%norm_r_list_1=HISTORY_1_128(:,1);
%norm_r_list_2=HISTORY_2_128(:,1);
