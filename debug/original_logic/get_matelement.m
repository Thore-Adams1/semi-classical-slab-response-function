function [A1,A2,G,H,arrayA1,arrayA2,arrayG,arrayH] = get_matelement(n,m,kvec,omega,tau,p,L,dtheta,dphi,vF)

% parameters;
% NOT ORIGNALLY DEFINED IN THE CODE
wp=1;
% 

wbar = omega + 1i/tau;

theta = 0:dtheta:pi/2;
phi = 0:dphi:2*pi;

arrayA1 = zeros(length(theta),length(phi));
arrayA2 = arrayA1;
arrayG = arrayA1;
arrayH = arrayA1;
for lt = 1:length(theta)
    for lp = 1:length(phi)
        velocity = vF*[sin(theta(lt))*cos(phi(lp)),sin(theta(lt))*sin(phi(lp)),cos(theta(lt))];
        [Atilde1,~,~,~] = get_Atilde1(n,m,kvec,wbar,velocity,p,L);
        [Atilde2,~,~] = get_Atilde2(n,m,kvec,wbar,velocity,p,L);
        [Gtilde,~,~] = get_Gtilde(n,m,kvec,wbar,velocity,p,L);
        [Htilde,~,~,~] = get_Htilde(n,m,kvec,wbar,velocity,p,L);
        arrayA1(lt,lp) = Atilde1*sin(theta(lt));
        arrayA2(lt,lp) = Atilde2*sin(theta(lt));
        arrayG(lt,lp) = Gtilde*sin(theta(lt));
        arrayH(lt,lp) = Htilde*sin(theta(lt));
    end
end

%
% lt = 5;lp = 5;
% disp("theta(lt)"+theta(lt))
% disp(theta)
% disp("phi(lp)"+phi(lp))
% disp(phi)
% velocity = vF*[sin(theta(lt))*cos(phi(lp)),sin(theta(lt))*sin(phi(lp)),cos(theta(lt))];
% % [Atilde1,~,~,~] = get_Atilde1(n,m,kvec,wbar,velocity,p,L);
% % arrayA1(lt,lp) = Atilde1*sin(theta(lt));
% % [Atilde2,~,~] = get_Atilde2(n,m,kvec,wbar,velocity,p,L);
% % arrayA2(lt,lp) = Atilde2*sin(theta(lt));
% % [Gtilde,~,~] = get_Gtilde(n,m,kvec,wbar,velocity,p,L);
% % arrayG(lt,lp) = Gtilde*sin(theta(lt));
% [Htilde,~,~,~] = get_Htilde(n,m,kvec,wbar,velocity,p,L);
% arrayH(lt,lp) = Htilde*sin(theta(lt));

%

Rcol = ones(length(phi),1);
Lrow = ones(1,length(theta));

A1 = dtheta*dphi*Lrow*arrayA1*Rcol;
A2 = dtheta*dphi*Lrow*arrayA2*Rcol;
G = dtheta*dphi*Lrow*arrayG*Rcol;
H = dtheta*dphi*Lrow*arrayH*Rcol;

Ln = L/(2-(n==0));
qm = m*pi/L;

fA = (1i*wbar/Ln)*(wp/vF)^2*(3/(4*pi)^2);
fGH = (1i*wbar/Ln)*(wp/vF)^2*(3/(4*pi)^2)*4*pi/(kvec*kvec'+qm^2);
% if (m == 0 & n == 6)
%     disp("dtheta" + dtheta)
%     disp("length(theta)" +  length(theta))
%     disp(theta)
%     disp("H"+H)
%     disp("HSUM"+Lrow*arrayH*Rcol)
%     disp("fA"+fA)
%     disp("fGH"+fGH)
%     disp("fA*A1"+fA*A1)
%     disp("fA*A2"+fA*A2)
%     disp("fGH*G"+fGH*G)
%     disp("fGH*H"+fGH*H)
% exit()
% end
   

A1 = fA*A1;
A2 = fA*A2;
G = fGH*G;
H = fGH*H;
end
% g = get_matelement(0,0,[0.01,0],0.5,1,1, 100, 10, 10,1)


