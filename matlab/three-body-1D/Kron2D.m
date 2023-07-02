classdef Kron2D
% implementation of the operator y=op*x with
% op=a1*(I_m (x) C_n) + a2*(D_m (x) I_n) + V
    properties
        n % dimension of C
        m % dimension of D
        a1 % scalar factor before term 1
        a2 % scalar factor before term 2
        C % dense matrix in term 1
        D % dense matrix in term 2
        V % potential matrix, n*m x n*m
    end

    methods
        function obj = Kron2D(a1,C,a2,D,V)
            obj.a1=a1;
            obj.a2=a2;
            obj.C=C;
            obj.D=D;
            obj.V=V;
            obj.n=size(C,1);
            obj.m=size(D,1);
        end
    
        % apply operator to a vector x (called if you write y=op*x)
        function y = mtimes(op,x)
            ncols = size(x,2);
            y = zeros(size(x));
            for k=1:ncols
                X = reshape(x(:,k),op.n,op.m);
                Y = reshape(op.a1*op.C*X+op.a2*X*op.D', op.n*op.m,1);
                y(:,k)=Y;
            end
            y = y + op.V.*x;
        end

        % apply transposed operator to a vector x
        function y = mTtimes(op,x)
            ncols = size(x,2);
            y = zeros(size(x));
            for k=1:ncols
                X = reshape(x(:,k),op.n,op.m);
                Y = reshape(op.a1*op.C'*X+op.a2*X*op.D, op.n*op.m,1);
                y(:,k)=Y;
            end
            y = y + op.V'.*x;
        end

        % approximate inverse operation, x = Op\b
        function x = mldivide(op,b)
            ncols = size(b,2);
            x = zeros(size(b));
            for k=1:ncols
                B = reshape(b(:,k),op.n,op.m);
                X = sylvester(op.a1*op.C,op.a2*op.D',B);
                x(:,k) = reshape(X,op.n*op.m,1);
            end
        end

        function sz = size(op,d)
            N = op.n*op.m;
            sz = [N,N];
            if exist('d')
                if (d>0 && d<=2)
                    sz=sz(d);
                end
            end
        end
       
    end
end
