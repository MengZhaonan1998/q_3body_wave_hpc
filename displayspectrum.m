

l = plot(real(Lambda),imag(Lambda),'b.',real(diag(Lambda_jdqz)),imag(diag(Lambda_jdqz)),'ro');
l(1).MarkerSize = 20; % set marker size of 8 for the first line (x1,y1)
l(2).MarkerSize = 10;
xlabel('real');
ylabel('imaginary');
legend('approximated by quadratic JD','approximated by JDQZ');
grid on;
