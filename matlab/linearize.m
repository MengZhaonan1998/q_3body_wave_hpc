% Linearization
function [A,B] = linearize(K,C,M)
% the first companion linearization
N = length(K);
I = speye(N);
Z = spalloc(N,N,0);

if issparse(K) == 0
    % non-sparse matrix
    Z = zeros(N);
    I = eye(N);
    A = [C,K;-I,Z];
    B = [-M,Z;Z,-I];
else
    % sparse matrix
    Z = spalloc(N,N,0);
    I = speye(N);
    A = [C,K;-I,Z];
    B = [-M,Z;Z,-I];
end
end

