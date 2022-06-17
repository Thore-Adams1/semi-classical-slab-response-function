function [Gtilde,Gtb,Gts]= get_Gtilde(n,m,kvec,wbar,velocity,p,L)

% Gtilde = \tilde{G}/2

% Gtb = \tilde{G}^b/2

% Gts = \tilde{G}^s*exp(i*\tilde{\omega})*L/vz)/2

% n, m are non-negative integers

% kvec = [kx,ky], row vector for wave vector

% wbar = \bar{\omega}

% velocity = [vx,vy,vz], row vector for velocity of electrons

% p = Fuchs parameter

% L = slab thickness

symmetry = (1+(-1)^(m+n))/2;

k = sqrt(kvec*kvec');

vz = velocity(3);

vpara = [velocity(1),velocity(2)];

kvdot = vpara*(kvec');

wtilde = wbar - kvdot;

phase = exp(1i*wtilde*L/vz);

D = 1/(1-p^2*phase^2);

qm = m*pi/L;

fac1 = symmetry*vz*((kvdot+qm*vz)/(wtilde-qm*vz) + (kvdot-qm*vz)/(wtilde+qm*vz));

A = (kvdot+1i*k*vz)/(wtilde - 1i*k*vz); 
B = (kvdot-1i*k*vz)/(wtilde + 1i*k*vz);

fac2 = symmetry*vz*(A + (-1)^m*exp(-k*L)*B);

fac3 = symmetry*vz*(B - A);

Gtb = (fac1 - fac2)*(1-(-1)^m*phase) - fac3*(1-(-1)^m*exp(-k*L));

Gts = fac1*(2*phase - (-1)^m*(phase^2 + 1)) ...
    + A*symmetry*vz*(exp(-k*L)-phase+(-1)^n*(phase-exp(-k*L))*phase)...
    + B*symmetry*vz*(phase^2*exp(-k*L)-phase+(-1)^n*(1-phase*exp(-k*L)));

Gtilde = Gtb + Gts*(D*p^2*phase + (-1)^n*D*p);

% disp("symmetry"+symmetry)
% disp("k"+k)
% disp("vz"+vz)
% disp("vpara"+vpara)
% disp("kvdot"+kvdot)
% disp("wtilde"+wtilde)
% disp("phase"+phase)
% disp("D"+D)
% disp("qm"+qm)
% disp("fac1"+fac1)
% disp("A"+A)
% disp("B"+B)
% disp("fac2"+fac2)
% disp("fac3"+fac3)
% disp("Gtb"+Gtb)
% disp("Gts"+Gts)
% disp("Gtilde"+Gtilde)
end