function [Atilde2,At2b,At2s] = get_Atilde2(n,m,kvec,wbar,velocity,p,L)

% Atilde2 = \tilde{A}^{(2)}

% At2b  = \tilde{A}^{2b}

% At2s = \tilde{A}^{2s}*exp(i\tilde{\omega}L/v_z)

% n, m are non-negative integers

% kvec = [kx,ky], row vector for wave vector

% wbar = \bar{\omega}

% velocity = [vx,vy,vz], row vector for velocity of electrons

% p = Fuchs parameter

% L = slab thickness

symmetry = (1+(-1)^(m+n))/2;

vz = velocity(3);

vpara = [velocity(1),velocity(2)];

kvdot = vpara*(kvec');

wtilde = wbar - kvdot;

phase = exp(1i*wtilde*L/vz);

D = 1/(1-p^2*phase^2);

qm = m*pi/L;

factor = symmetry*vz*((kvdot+qm*vz)/(wtilde-qm*vz) + (kvdot-qm*vz)/(wtilde+qm*vz));

At2b = (1-(-1)^m*phase)*factor;

At2s = (2*phase - (-1)^m*(phase^2+1))*factor;

Atilde2 = At2b + At2s*(D*p^2*phase+(-1)^n*D*p);

% disp("symmetry"+symmetry);
% disp("vz"+vz);
% disp("vpara"+vpara);
% disp("kvdot"+kvdot);
% disp("wtilde"+wtilde);
% disp("phase"+phase);
% disp("D"+D);
% disp("qm"+qm);
% disp("factor"+factor);
% disp("At2b"+At2b);
% disp("At2s"+At2s);
% disp("Atilde2"+Atilde2);

end