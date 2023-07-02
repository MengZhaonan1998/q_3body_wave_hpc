classdef diagOp
    properties
        diag_entry
    end
    methods
        function obj = diagOp(diag_entry)
            obj.diag_entry = diag_entry;
        end

        function y=mtimes(op,x)
            ncols = size(x,2);
            y = zeros(size(x));
            for k=1:ncols
                y(:,k)=op.diag_entry * x(:,k);
            end
        end
    end

end