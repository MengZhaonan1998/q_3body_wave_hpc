function [theta,c]=linearization(K,C,M)
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

end