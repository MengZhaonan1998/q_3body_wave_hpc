function [V,R]=RepGS(Z,V,gamma)
%
% Orthonormalisation using repeated Gram-Schmidt
%   with the Daniel-Gragg-Kaufman-Stewart (DGKS) criterion
%
% Q=RepGS(V)
%   The n by k matrix V is orthonormalized, that is,
%   Q is an n by k orthonormal matrix and 
%   the columns of Q span the same space as the columns of V
%   (in fact the first j columns of Q span the same space
%   as the first j columns of V for all j <= k).
%
% Q=RepGS(Z,V)
%  Assuming Z is n by l orthonormal, V is orthonormalized against Z:
%  [Z,Q]=RepGS([Z,V])
%
% Q=RepGS(Z,V,gamma)
%  With gamma=0, V is only orthogonalized against Z
%  Default gamma=1 (the same as Q=RepGS(Z,V))
%
% [Q,R]=RepGS(Z,V,gamma)
%  if gamma == 1, V=[Z,Q]*R; else, V=Z*R+Q; end
 
% coded by Gerard Sleijpen, March, 2002

if nargin == 1, V=Z; Z=zeros(size(V,1),0); end   
if nargin <3, gamma=1; end

[n,dv]=size(V); [m,dz]=size(Z);

if gamma, l0=min(dv+dz,n); else, l0=dz; end
R=zeros(l0,dv); 

if dv==0, return, end
if dz==0 & gamma==0, return, end

% if m~=n
%   if m<n, Z=[Z;zeros(n-m,dz)]; end
%   if m>n, V=[V;zeros(m-n,dv)]; n=m; end
% end

if (dz==0 & gamma)
   j=1; l=1; J=1;
   q=V(:,1); nr=norm(q); R(1,1)=nr;
   while nr==0, q=rand(n,1); nr=norm(q); end, V(:,1)=q/nr;
   if dv==1, return, end    
else
   j=0; l=0; J=[];
end

while j<dv,
   j=j+1; q=V(:,j); nr_o=norm(q); nr=eps*nr_o;
   if dz>0, yz=Z'*q;     q=q-Z*yz;      end
   if l>0,  y=V(:,J)'*q; q=q-V(:,J)*y;  end
   nr_n=norm(q);
  
   while (nr_n<0.5*nr_o & nr_n > nr)
      if dz>0, sz=Z'*q;     q=q-Z*sz;     yz=yz+sz; end
      if l>0,  s=V(:,J)'*q; q=q-V(:,J)*s; y=y+s;    end
      nr_o=nr_n; nr_n=norm(q);                
   end
   if dz>0, R(1:dz,j)=yz; end
   if l>0,  R(dz+J,j)=y;  end

   if ~gamma
     V(:,j)=q;
   elseif l+dz<n, l=l+1; 
     if nr_n <= nr % expand with a random vector
       % if nr_n==0
       V(:,l)=RepGS([Z,V(:,J)],rand(n,1));
       % else % which can be numerical noice
       %   V(:,l)=q/nr_n;
       % end
     else
       V(:,l)=q/nr_n; R(dz+l,j)=nr_n; 
     end
     J=[1:l];
   end 

end % while j

if gamma & l<dv, V=V(:,J); end

end