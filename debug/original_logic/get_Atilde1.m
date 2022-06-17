function [Atilde1,At1b,At1s1,At1s2] = get_Atilde1(n,m,kvec,wbar,velocity,p,L)

% Atilde1 = \tilde{A}^{(1)}

% At1b = \tilde{A}^{1b}

% At1s1 = \tilde{A}^{1s(1)}*exp(2i\tilde{\omega}L/v_z)

% At1s2 = \tilde{A}^{1s(2)}

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

Ln = L/(2-(n==0));

qm = m*pi/L;

qn = n*pi/L;

phase = exp(1i*wtilde*L/vz);

D = 1/(1-p^2*phase^2);

factor1 = (kvdot+qm*vz)^2/(wtilde-qm*vz) + (kvdot-qm*vz)^2/(wtilde+qm*vz);

At1b = (n==m)*(Ln/1i)*factor1 - symmetry*(1-(-1)^m*phase) ... 
    *(wtilde*vz/(wtilde^2-qn^2*vz^2))*(factor1 + kvdot);

factor2 = 2*wbar*wtilde*vz*(wtilde*kvdot+qm^2*vz^2)/((wtilde^2-qn^2*vz^2)*(wtilde^2-qm^2*vz^2));

At1s1 = - symmetry*factor2*(2*phase^2 - (-1)^m*(phase^3+phase));

At1s2 = symmetry*(1-(-1)^n*phase)*(1-(-1)^m*phase);

Atilde1 = At1b + D*p^2*At1s1 + D*p*At1s2;

end