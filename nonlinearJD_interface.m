opt.pot_type = 'G';
opt.v0 = 1;
opt.q = 3;
[K,C,M] = b2d1_operator(100,opt);
P.A0 = K;
P.A1 = C;
P.A2 = M;

opts.mindim=10;
opts.maxdim=30;
opts.tol=1e-9;
opts.maxit=300;
opts.lsolver='exact';
[Lambda,X] = jdpoly(P,10,0+0i,opts);

figure;

l = plot(real(Lambda),imag(Lambda),'b.',real(diag(Lambda_jdqz)),imag(diag(Lambda_jdqz)),'ro');
l(1).MarkerSize = 20; % set marker size of 8 for the first line (x1,y1)
l(2).MarkerSize = 10;
xlabel('real');
ylabel('imaginary');
legend('approximated by quadratic JD','approximated by JDQZ');
grid on;


