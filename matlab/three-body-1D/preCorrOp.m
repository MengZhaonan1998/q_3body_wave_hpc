classdef preCorrOp
    properties
        n              % problem size
        Kop       % tensor K
        Cop       % tensor C
        Mop       % tensor M 
        precondOp % tensor precond
        theta_best     % ritz value          
        u              % ritz vector 
        w              % ..
        mu             % ..
        w_tilde        %..
    end

    methods
        function obj = preCorrOp(Kop, Cop, Mop, theta_best, precondOp, u, w, w_tilde, mu)
            obj.Kop = Kop;
            obj.Cop = Cop;
            obj.Mop = Mop;
            obj.precondOp = precondOp;
            obj.theta_best = theta_best;
            obj.u = u;
            obj.w = w;
            obj.w_tilde = w_tilde;
            obj.mu = mu;
            obj.n = size(u,1);
        end

        % apply correction operator (gmres)
        function z = correctapply(op, p)
            T = mtimes(op.Kop,p) + op.theta_best * mtimes(op.Cop,p) + op.theta_best * op.theta_best * mtimes(op.Mop,p); 
            y = (eye(op.n) - (op.w*op.u')/(op.u'*op.w)) * T;
            y_tilde = mldivide(op.precondOp,y);
            z = y_tilde - op.u' * y_tilde * op.w_tilde / op.mu;
        end
    end
end