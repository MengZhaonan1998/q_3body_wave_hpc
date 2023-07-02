classdef correctOp
    properties
        n    % problem size
        Kop  % tensor K
        Cop  % tensor C
        Mop  % tensor M
        u      % .. 
        w      % ..
        theta  % ..
    end

    methods
        function obj = correctOp(Kop, Cop, Mop, u, w, theta)
            obj.Kop = Kop;
            obj.Cop = Cop;
            obj.Mop = Mop;
            obj.u = u;
            obj.w = w;
            obj.theta = theta;
            obj.n = size(u,1);
        end

        % apply correction operator (gmres)
        function y = correctapply(op, x)
            ncols = size(x,2);
            y = zeros(size(x));
            for k=1:ncols
                dux = op.u' * x(:,k); % allreduce dot
                y(:,k) = x(:,k) - op.u * dux / (op.u'*op.u); % axpby
                
                mv = op.theta * op.theta * mtimes(op.Mop, y(:,k));  % tensor prod
                cv = op.theta * mtimes(op.Cop, y(:,k));    % tensor prod axpby
                kv = mtimes(op.Kop, y(:,k));               % tensor prod axpby  
                y(:,k) = kv+cv+mv;
                
                duy = op.u' * y(:,k);    % allreduce dot
                duw = op.u' * op.w;      % allreduce dot
                y(:,k) = y(:,k) - (duy/duw) * op.w;  % axpby
            end
        end
    end
end