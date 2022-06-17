function [chip,chim,epsp,epsm,chipb,chips,chimb,chims,Hp,Gp,Hm,Gm] = get_chi(kvec,omega,wbar,lc)

% This calculates the density-density response function \chi

% l and lp index the elements of \chi

% chip = chi_+, chim = chi_-

% lc is the wave number cutoff, lc ~ k*L/2*pi

Zm = ones(lc+1,1); % column vector

Zp = Zm; 
Zp(1,1) = 1/2;

Hp = zeros(lc+1,lc+1); 
Hm = Hp; 
Ap = Hp; 
Am = Hp; 
gp = Hp; 
gm = Hp;
Gp = zeros(1,lc+1); % row vector
Gm = Gp;

for l=1:lc+1
    for lp=1:lc+1
        [A1,A2,G,H,~,~,~,~] = get_matelement(2*(l-1),2*(lp-1),kvec,omega);
        Hp(l,lp) = H; Ap(l,lp) = A1 + A2; gp(l,lp) = G; Gp(1,lp) = G/Zp(l,1);
        [A1,A2,G,H,~,~,~,~] = get_matelement(2*l-1,2*lp-1,kvec,omega);
        Hm(l,lp) = H; Am(l,lp) = A1 + A2; gm(l,lp) = G; Gm(1,lp) = G/Zm(l,1);
    end
end

Iden = eye(lc+1,lc+1);

Hinvp = Iden/(Iden*wbar^2 - Hp);
Hinvm = Iden/(Iden*wbar^2 - Hm);

epsp = 1 - Gp*Hinvp*Zp; % The poles of this function give symmetric SPWs.
epsm = 1 - Gm*Hinvm*Zm; % The poles of this function give anti-symmetric SPWs.

chipb = Hinvp*Ap; % This determines the bulk response for the symmetric sector
chimb = Hinvm*Am; % This determines the bulk response for the anti-symmetric sector

chips = (1/epsp)*(Hinvp*Zp)*(Gp*Hinvp*Ap); % This determines the surface response for the symmetric sector
chims = (1/epsm)*(Hinvm*Zm)*(Gm*Hinvm*Am); % This determines the surface response for the symmetric sector

chip = chipb + chips;
chim = chimb + chims;

end



