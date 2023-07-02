function [Vsp] = evalpot(nR,nr,xR,xr,potopt)
V12=potopt.V12;
V13=potopt.V13;
V23=potopt.V23;
q = potopt.q;
if strcmp(potopt.pot_type, 'G')
    Vsp = buildGaussianPotential_3B_1D(nR,nr,xR,xr,V12,V13,V23);
elseif strcmp(potopt.pot_type, 'L') 
    Vsp = buildLorentzianPotential_3B_1D(nR,nr,xR,xr,V12,V13,V23,q);
else 
    error('Unknown potential type');
end
end

function [Vsp] = buildGaussianPotential_3B_1D(nR,nr,xR,xr,V12,V13,V23)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    temp=zeros(nR*nr,1);
    for i=1:nR
        for j=1:nr        
            val_a=xR(i).^2;
            val_b=0.25*xR(i).^2+xr(j).^2+xR(i).*xr(j);
            val_c=0.25*xR(i).^2+xr(j).^2-xR(i).*xr(j);
            temp((i-1)*nr+j)=-(V12*exp(-val_a)+V13*exp(-val_b)+V23*exp(-val_c));      
        end
    end
    Vsp=temp;
end

function [Vsp] = buildLorentzianPotential_3B_1D(nR,nr,xR,xr,V12,V13,V23,q)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    temp=zeros(nR*nr,1);
    for i=1:nR
        for j=1:nr
            val_a=xR(i).^2;
            val_b=0.25*xR(i).^2+xr(j).^2+xR(i)*xr(j);
            val_c=0.25*xR(i).^2+xr(j).^2-xR(i)*xr(j);
            temp((i-1)*nr+j)=-(V12*1/(1+val_a^q)+V13*1/(1+val_b^q)+V23*1/(1+val_c^q));
        end
    end
    Vsp=temp;
end