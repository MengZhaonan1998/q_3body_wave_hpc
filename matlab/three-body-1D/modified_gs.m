function Q=modified_gs(X,n)
for j=1:n
    V(:,j)=X(:,j);
end
for j=1:n
    Q(:,j)=V(:,j)/norm(V(:,j));
    for k=j+1:n
        V(:,k)=V(:,k)-Q(:,j)'*V(:,k)*Q(:,j);
    end
end
end