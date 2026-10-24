%% Water Steepest Descent w/ MATLAB Gradient & Symbolic Math
% Author: Jevin Xu
% Description: Stores cooling data. Estimates initial and environmental
%              temperature with steepest descent on least squares error.
%              Gradient found directly with MATLAB 'gradient' function.
%              Plots resulting model with observed data.
clear; clc; close;

syms r T0 Te

% Newton's Law of Cooling Model
T = @(t, r, T0, Te) Te + (T0 - Te).*exp((-t.*r.^2)./100);

% Parameters (a: initial step size; e: min gradient two-norm)
a = 1e-6;
e = 1.00e-6;

% Initialize a vector (iterative step size)
A(1) = a;

% Time vector t
t = [0:1:30]';

% Measured temperature values
Tm = [
212;
203;
195.6;
190.4;
184.6;
180.3;
176;
171.5;
168.6;
165;
162;
159.3;
156.9;
154.4;
151.5;
149.9;
147.9;
146.1;
144;
142.2;
140.4;
138.9;
137.5;
135.7;
134.4;
132.8;
131.4;
130.1;
129;
127.9;
126.7;
116.5;
117.3;
112.8;
109.8;
106.9;
104.6;
];

% Function to minimize (Least Squares Error = LSERR)
l = 1:31;
LSERRsym = sum((T(t(l), r, T0, Te) - Tm(l)).^2);
LSERR = matlabFunction(LSERRsym, 'Vars', {r, T0, Te});

% Vector of independent variables
v = [r; T0; Te];

% Initialization and initial guesses for parameters (r, T0, Te)
k = 1;
V(1:3, k) = [4, 212, 70]'; 

% Symbolic and callable version of gradient (in terms of r, T0, Te)
gsymb = gradient(LSERRsym, v);
g = matlabFunction(gsymb, 'Vars', {r, T0, Te});

% Run when gradient is not within 'e' of zero
while (norm(g(V(1,k), V(2,k), V(3,k))) > e && k < 1e5)
    % index j & define gradient
    j = 1;
    tempg = g(V(1,k), V(2,k), V(3,k));
    % Find right step size A(j)
    while (LSERR(V(1,k) - A(j) * tempg(1), V(2,k) - A(j) * tempg(2), V(3,k) - A(j) * tempg(3)) >= LSERR(V(1,k),V(2,k),V(3,k)) && j < 40)
        j = j + 1;
        A(j) = A(j-1)/2;
    end
    k = k + 1;
    V(1:3, k) = V(1:3, k-1) - A(j) * tempg;
end

% Print out optimized parameters (r, T0, Te)
disp(V(1:3, k))

% Plot over actual measurements (dt are small values of t to plot)
dt = [0:0.05:30]';


% Initialize T for computational efficiency (no need for continuous
% expansion)
dT = zeros(601, 1);
for z = 1:601
    dT(z) = T(dt(z), V(1,k), V(2,k), V(3,k));
end
plot(t,Tm(1:31), 'b', LineWidth=2, DisplayName='Actual');
hold on
plot(dt, dT, 'r', LineWidth=2, DisplayName='Approximation');
title('Water Temperature Over Time')
xlabel('Time(min)')
ylabel('Temperature(F)')
legend


