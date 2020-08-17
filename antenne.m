N=5;
l=0.3;
d = l/2;

theta = 0/180*pi;
dphi = 2*pi*d*sin(theta)/l;

n = (0:N-1)';
s = exp(j*dphi*n);

y = s;


theta1 = 20/180*pi;
dphi1 = 2*pi*d*sin(theta1)/l;
s1 = exp(j*dphi1*n);
theta2 = 40/180*pi;
dphi2 = 2*pi*d*sin(theta2)/l;
s2 = exp(j*dphi2*n);

%R = s1*s1' + s2*s2' + 0.001*eye(N);
R = zeros(N);
K=1000;
for k=1:K
    a1 = randn(1) + j * randn(1);
    a2 = randn(1) + j * randn(1);
    nn = a1 *s1 + a2 * s2 + 0.001*(randn(N,1)+j*randn(N,1));
    R = R + nn * nn';
end
R = R/K;

y = s + s1 + s2;

theta = [-90:90]/180*pi;
for k=1:length(theta)
    dphi = 2*pi*d*sin(theta(k) )/l;
    s = exp(j*dphi*n);
    w = inv(R)*s;
    T(k) = abs(w'*y)^2;
    
    SNR(k) = s'*inv(R)*s;
end
polar(theta,T)

figure
plot(theta/pi*180, 10*log10( SNR ));