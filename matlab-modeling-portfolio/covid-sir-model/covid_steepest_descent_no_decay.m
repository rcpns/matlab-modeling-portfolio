%% COVID-19 Steepest Descent Without Beta Decay
% Authors: Jevin Xu & Matthew Delaney
% Description: Stores weekly COVID data. Uses training data to find 
%              least squares error. Uses steepest descent and RK4 to 
%              estimate beta (transmission rate) and gamma (removal rate) 
%              and plots model with training data. 
clear; clc; close;

% Population
N = 3.3*10^8;

% Actual covid data
C_raw = [7; 9; 20; 24; 40; 73; 464; 3929; 31415; 126309; 302460; 521632; 717782; 922947; 1121383; 1303629; 1472661; 1638986; 1785422; 1935202; 2087078; 2267099; 2525377; 2867543; 3259748; 3726015; 4193916; 4625166; 5001075; 5363737; 5679525; 5974471; 6269724; 6515345; 6796322; 7101096; 7399444; 7733693; 8121327; 8570765; 9130159; 9920253; 10925098; 12100581; 13262982; 14543914; 16012396; 17530324; 18890446; 20271441; 21938592; 23528547; 24794408; 25863033];
C = C_raw / N;

% Number of data points
n = 54;

% h = step size; epsilon = tolerance for gradient;k = placeholder
h = 1;
epsilon = 1e-7;
k = 1;

% Initial step size
B(1) = 0.001;

% Initial guess for model parameters beta, gamma
A(:,1) = [0.4, 0.1]';

% Initial SIR vector Y0
Y0 = [1-C(1); C(1); 0];

% Time/index vector for real data
t = 1:1:54;

% SIR Model definition (Susceptible, Infectious, Recovered)
dS = @(S, I, beta) -(beta*S*I);
dI = @(S, I, beta, gamma) (beta*S*I) - gamma*I;
dR = @(I, gamma) gamma*I;

% Vector function of SIR rates, where Y is [S;I;R]
dY = @(t, Y, beta, gamma) [
dS(Y(1), Y(2), beta);
dI(Y(1), Y(2), beta, gamma);
dR(Y(2), gamma)
];

% Steepest descent 
while (norm(GradLS(dY, 54, Y0, h, C, A(1,k), A(2,k))) > epsilon && k < 500)
    index = 1;
    tempgrad = GradLS(dY, 54, Y0, h, C, A(1,k), A(2,k));
    % Find the right step size
    while (LSERR(dY, 54, Y0, h, C, A(1,k) - B(index) * tempgrad(1), A(2,k) - B(index) * tempgrad(2)) > LSERR(dY, 54, Y0, h, C, A(1,k), A(2,k)))
        index = index + 1;
        B(index) = B(index - 1)/2;
    end
    k = k + 1;
    A(:, k) = A(:, k-1) - B(index) * tempgrad;
    fprintf('Iter %d: Beta=%.4f, Gamma=%.4f\n', k, A(1,k), A(2,k));
    % Constraint: Don't let Beta or Gamma go below 0.0001
    if A(1, k) < 0, A(1, k) = 0.0001; end
    if A(2, k) < 0, A(2, k) = 0.0001; end
end
% Plotting comparison
R_final = rk4(dY, 54, Y0, h, A(1,k), A(2,k)); 
plot(1:54, C_raw, 'ro-', 1:54, (R_final(2,:) + R_final(3,:)) * N, 'b-', 'LineWidth', 2) ;legend('Actual Data', 'SIR Model Prediction');
title('Total Cases: SIR Model vs. Actual');
xlabel('Weeks'); ylabel('Total Cases');


% Functions:
% RK4 (dY is a function, Y0 is a 3x1 vector)
% Y = rk4(dY, t, Y0, h, beta, gamma) gives Y, approximated values for SIR up to t
function output = rk4(dY, tEnd, Y0, h, beta, gamma)
    t = 1:h:tEnd;
    n = length(t);
    output = zeros(3,n);
    output(:,1) = Y0;
    for i = 1:n-1
        k1 = h*dY(t(i), output(:,i), beta, gamma);
        k2 = h*dY(t(i) + (h/2), output(:,i) + (k1/2), beta, gamma);
        k3 = h*dY(t(i) + (h/2), output(:,i) + (k2/2), beta, gamma);
        k4 = h*dY(t(i) + h, output(:,i) + k3, beta, gamma);
        output(:,i+1) = output(:,i) + (k1 + 2*k2 + 2*k3 + k4)/6;
    end
end

% LSERR: Least Squares Error Function 
% LSERR(dY, t, Y0, h, C, beta, gamma) gives the LS error of using beta 
% and gamma
function output = LSERR(dY, t, Y0, h, C, beta, gamma)
    % R is resulting 3xn matrix of rk4 function at each 
    R = rk4(dY, t, Y0, h, beta, gamma);
    output = sum(((R(2,:) + R(3,:) - C')).^2);
end

% GradLS: Estimation for gradient of LSERR
% GradLS(dY, t, Y0, h, C, beta, gamma) outputs gradient of LSERR in the
% order of beta, gamma
function output = GradLS(dY, t, Y0, h, C, beta, gamma)
    % e here is the step size to estimate derivatives
    step = 1e-6;
    Lb1 = LSERR(dY, t, Y0, h, C, beta - (step/2), gamma);
    Lb2 = LSERR(dY, t, Y0, h, C, beta + (step/2), gamma);
    Lg1 = LSERR(dY, t, Y0, h, C, beta, gamma - (step/2));
    Lg2 = LSERR(dY, t, Y0, h, C, beta, gamma + (step/2));
    output = [
    (Lb2 - Lb1)/step;
    (Lg2 - Lg1)/step;
    ];
end
