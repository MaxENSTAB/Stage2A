function adaptant

l = 0.3; %Lambda
d = l/2;
N = 5; % Number of sensors (must be uneven)
dphi = 2*pi*d/l;

% Electronic antenna, steering weight given by s0
s0 = getsteeringvector(N, dphi, 0);
plotdiagram(N, dphi, s0)

% Adaptive antenna, steering weight given by s0, interference in s1
s1 = getsteeringvector(N, dphi, 30/180*pi);
R = s1*s1' + 0.01*eye(N);
w = inv(R)*s0;
plotdiagram(N, dphi, w)


% Adaptive antenna, steering weight given by s0, interference in s1 and s2, uncorrelated
s2 = getsteeringvector(N, dphi, 50/180*pi);

L = 1000;
phi1 = randn(L,1)+j*randn(L,1);
phi2 = randn(L,1)+j*randn(L,1);
sn = s1*phi1' + s2*phi2';
R = sn*sn'+0.01*eye(N);
w = inv(R)*s0;
plotdiagram(N, dphi, w)

plotsnr(N, dphi, R)


%+++++++++++++++++++++++++++++++++++++++++++++++++++
function plotdiagram(N, dphi, s0 )
theta = (-90:90)'/180*pi;
A = zeros(length(theta),1);
for k = 1:length(theta)
    s = getsteeringvector(N, dphi, theta(k));
    A(k) = s0'*s;
end
plot( theta/pi*180, abs(A));
figure
polar( theta, abs(A));

%+++++++++++++++++++++++++++++++++++++++++++++++++++
function plotsnr(N , dphi,  R )
theta = (-90:90)'/180*pi;
A=zeros(length(theta),1);
for k=1:length(theta)
    s=getsteeringvector( N, dphi, theta(k));
    A(k) = s'*inv(R)*s;
end
plot( theta/pi*180, 20*log10(abs(A)));
figure
polar( theta, abs(A));

%+++++++++++++++++++++++++++++++++++++++++++++++++++
function s = getsteeringvector(N, dphi, theta)

n = (1:N)';
s = exp(j*n*dphi*sin(theta));
