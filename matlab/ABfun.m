function out=ABfun(varargin)

n=100;

if nargin>1
  flag=varargin{2};
  switch flag
  case 'dimension'
    out = n; return
  case 'A'
    out = Af(n,varargin{:}); return
  case 'B'
    out = Bf(varargin{1},n); return
  otherwise, error(['unknown argument'])
  end
end

return

