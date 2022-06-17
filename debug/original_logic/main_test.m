clear;

% n = 1;
% m = 1;
% kvec = [0.1,0];
% velocity = [0.6,0.5,0.6];
% wbar = 0.5+1i/10;
% L = 10;
% p = 1;
% [Atilde2,At2b,At2s] = get_Atilde2(n,m,kvec,wbar,velocity,p,L)
% [Gtilde,Gtb,Gts] = get_Gtilde(n,m,kvec,wbar,velocity,p,L)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% n=1
% m=1
% 
% kvec = [0.1,0];
% omega = 0.5;
% 
% parameters;
% 
% theta = 0:dtheta:pi/2;
% phi = 0:dphi:2*pi;
% 
% tic
% [A1,A2,G,H,arrayA1,arrayA2,arrayG,arrayH] = get_matelement(n,m,kvec,omega);
% toc
% 
% A1
% A2
% G
% H

% surf(theta,phi,real(arrayH))
% view(2)
% shading interp
% title(['Real(Htilde):','\omega=',num2str(omega),'--k=',num2str(kvec(1)),'--n=',num2str(n),'--m=',num2str(m)])
% xlabel('\theta')
% ylabel('\phi')
% xlim([-0.002,pi/2])
% ylim([0,2*pi])
% colorbar
% annotation('textbox', [0.5, 0.2, 0.2, 0.1], 'String', ["p=",num2str(p)])
% 
% figure
% surf(theta,phi,real(arrayG))
% view(2)
% shading interp
% title(['Real(Gtilde):','\omega=',num2str(omega),'--k=',num2str(kvec(1)),'--n=',num2str(n),'--m=',num2str(m)])
% xlabel('\theta')
% ylabel('\phi')
% xlim([-0.002,pi/2])
% ylim([0,2*pi])
% colorbar
% annotation('textbox', [0.5, 0.2, 0.2, 0.1], 'String', ["p=",num2str(p)])
% 
% figure
% surf(theta,phi,real(arrayA1))
% view(2)
% shading interp
% title(['Real(A1tilde):','\omega=',num2str(omega),'--k=',num2str(kvec(1)),'--n=',num2str(n),'--m=',num2str(m)])
% xlabel('\theta')
% ylabel('\phi')
% xlim([-0.002,pi/2])
% ylim([0,2*pi])
% colorbar
% annotation('textbox', [0.5, 0.2, 0.2, 0.1], 'String', ["p=",num2str(p)])
% 
% figure
% surf(theta,phi,real(arrayA2))
% view(2)
% shading interp
% title(['Real(A2tilde):','\omega=',num2str(omega),'--k=',num2str(kvec(1)),'--n=',num2str(n),'--m=',num2str(m)])
% xlabel('\theta')
% ylabel('\phi')
% xlim([-0.002,pi/2])
% ylim([0,2*pi])
% colorbar
% annotation('textbox', [0.5, 0.2, 0.2, 0.1], 'String', ["p=",num2str(p)])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% kvec = [0.1,0];
% omega = 0.7:0.03:1;
% 
% parameters;
% 
% lc = ceil(1*sqrt(kvec*kvec')*L/(2*pi))
% 
% fp = zeros(1,length(omega));
% fm = fp;
% for lw = 1:length(omega)
%     tic
%     [~,~,epsp,epsm,~,~,~,~,~,~,~,~] = get_chi(kvec,omega(lw),omega(lw)+1i/tau,lc);
%     toc
%     fp(1,lw) = abs(epsp); fm(1,lw) = abs(epsp);
% end
% 
% plot(omega,1./fp^2,'r',omega,1./fm^2,'b')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

kvec = [0.01,0];
omega = 0.91;

parameters;

lc = 20% ceil(2*8*sqrt(kvec*kvec')*L/(2*pi)) % 102

tic
[chip,chim,epsp,epsm,chipb,chips,chimb,chims,Hp,Gp,Hm,Gm] = get_chi(kvec,omega,omega+1i/tau,lc);
toc

1/abs(epsp)^2

1/abs(epsm)^2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% k = 0.01;
% 
% z = 0.03;
% 
% L = 1;
% 
% lc = ceil(5*2*k*L/(2*pi))
% 
% l = 0:lc;
% 
% qp = (2*pi*l/L);
% qm = (pi/L)*(2*l+1);
% 
% Lp = L./(2-(l==0));
% Lm = L/2;
% 
% (1-exp(-k*L))*k*sum(cos(qp*z)./(Lp.*(k^2+qp.^2)))...
%     +  (1+exp(-k*L))*k*sum(cos(qm*z)./(Lm.*(k^2+qm.^2)))
% 
% exp(-k*z)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





















