function [Htilde,Htb,Hts1,Hts2] = get_Htilde(n,m,kvec,wbar,velocity,p,L)

% Htilde = \tilde{H}/2

% Htb = \tilde{H}^b/2

% Hts1 = \tilde{H}^{s(1)}*exp(2i\tilde{\omega}L/v_z)/2

% Hts2 = \tilde{H}^{s(2)}/2

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

qm = m*pi/L;

qn = n*pi/L;

Ln = L/(2-(n==0));
phase = exp(1i*wtilde*L/vz);
% xxx = (1i*wtilde*L/vz);
% eee = exp(xxx);
% disp("CURRENTPHASE"+phase);
% disp("CONTENTS"+ xxx)
% u = 1i*wtilde*L
% disp("CURRENTU"+u);
% disp("EEE"+ eee)
D = 1/(1-p^2*phase^2);

Phim0 = 1 - (-1)^m*exp(-k*L);

factor1 = (kvdot+qm*vz)^2/(wtilde-qm*vz) + (kvdot-qm*vz)^2/(wtilde+qm*vz);

factor2 = (kvdot+1i*k*vz)^2/(wtilde-1i*k*vz) + (kvdot-1i*k*vz)^2/(wtilde+1i*k*vz);

Htb = (n==m)*(Ln/1i)*factor1 ...
    - symmetry*(1-(-1)^m*phase)*(wtilde*vz/(wtilde^2-qn^2*vz^2))...
    *(factor1 - (kvdot+1i*k*vz)^2/(wtilde-1i*k*vz) - (-1)^m*exp(-k*L)*(kvdot-1i*k*vz)^2/(wtilde+1i*k*vz) + Phim0*(kvdot-1i*k*vz))...
    + symmetry*(1i*k*Phim0/(k^2+qn^2))*factor2;

% h1 = (n==m)*(Ln/1i)*factor1;
% h2 = symmetry*(1-(-1)^m*phase)*(wtilde*vz/(wtilde^2-qn^2*vz^2));
% h3 = (factor1 - (kvdot+1i*k*vz)^2/(wtilde-1i*k*vz) - (-1)^m*exp(-k*L)*(kvdot-1i*k*vz)^2/(wtilde+1i*k*vz) + Phim0*(kvdot-1i*k*vz));
% h4 = symmetry*(1i*k*Phim0/(k^2+qn^2))*factor2;
% h5 = (1i*k*Phim0/(k^2+qn^2))

fac1 = 2*(wtilde*kvdot+qm^2*vz^2)/((wtilde^2-qm^2*vz^2)*(wtilde^2-qn^2*vz^2));

fac2 = (kvdot+1i*k*vz)/((wtilde^2-qn^2*vz^2)*(wtilde-1i*k*vz));

fac3 = (kvdot-1i*k*vz)/((wtilde^2-qn^2*vz^2)*(wtilde+1i*k*vz));

Hts1 = -symmetry*wbar*wtilde*vz*(fac1*(2*phase^2-(-1)^m*(phase^3+phase))...
    +fac2*(phase*exp(-k*L)-phase^2+(-1)^m*(phase-exp(-k*L))*phase^2)...
    +fac3*(phase^3*exp(-k*L)-phase^2+(-1)^m*(phase-phase^2*exp(-k*L))));

Hts2 = symmetry*wbar*wtilde*vz*(fac1*(1-(-1)^n*phase)*(1-(-1)^m*phase)...
    -(-1)^m*fac2*(1-(-1)^n*phase)*(exp(-k*L)-phase)...
    -fac3*(1-(-1)^n*phase)*(1-phase*exp(-k*L)));

% x123 = (fac1*(1-(-1)^n*phase)*(1-(-1)^m*phase)...
% -(-1)^m*fac2*(1-(-1)^n*phase)*(exp(-k*L)-phase)...
% -fac3*(1-(-1)^n*phase)*(1-phase*exp(-k*L)));
% x1 = symmetry*wbar*wtilde*vz;
% x2 = fac1*(1-(-1)^n*phase)*(1-(-1)^m*phase);
% x3_1 = (-1)^m*fac2;
% x3 = (-1)^m*fac2*(1-(-1)^n*phase)*(exp(-k*L)-phase);
% x4 = fac3*(1-(-1)^n*phase)*(1-phase*exp(-k*L));
Htilde = Htb + D*p^2*Hts1 + D*p*Hts2;
% 1j + (0.5632+0.01j) *1000 / 6.1232e-17
% disp("___________________________"+m+"_"+n);
% disp("symmetry"+symmetry);
% disp("k"+k);
% disp("vz"+vz);
% disp("vpara"+vpara);
% disp("kvdot"+kvdot);
% disp("wtilde"+wtilde);
% disp("qm"+qm);
% disp("qn"+qn);
% disp("Ln"+Ln);
% disp("x123"+ x123)
% disp("x1"+ x1)
% disp("x2"+ x2)
% disp("x3_1"+ x3_1)
% disp("x3"+ x3)
% disp("x4"+ x4)
% disp("phase"+phase);
% disp("D"+D);
% disp("Phim0"+Phim0);
% disp("factor1"+factor1);
% disp("factor2"+factor2);
% disp("h1"+h1)
% disp("h2"+h2)
% disp("h3"+h3)
% disp("h4"+h4)
% disp("h5"+h5)
% disp("Htb"+Htb);
% disp("fac1"+fac1);
% disp("fac2"+fac2);
% disp("fac3"+fac3);
% disp("Hts1"+Hts1);
% disp("Hts2"+Hts2);
% disp("Htilde"+Htilde);

end