%% COVID-19 SIR Model With 'fminsearch' and Beta Decay
% Author: Jevin Xu
% Description: Stores weekly COVID data. Uses training data to find 
%              least squares error. Uses 'fminsearch' and
%              'ode45' to estimate beta (transmission rate), gamma
%              (removal rate), beta_0 (initial beta), and d (beta decay
%              rate) and fit to training data. Plots resulting model
%              along with training and validation data.
clear; clc; close;

% Training data (first 54 Days, used to find beta/gamma, no vaccine)
C_train = [7; 9; 20; 24; 40; 73; 464; 3929; 31415; 126309; 302460; 521632; 717782; 922947; 1121383; 1303629; 1472661; 1638986; 1785422; 1935202; 2087078; 2267099; 2525377; 2867543; 3259748; 3726015; 4193916; 4625166; 5001075; 5363737; 5679525; 5974471; 6269724; 6515345; 6796322; 7101096; 7399444; 7733693; 8121327; 8570765; 9130159; 9920253; 10925098; 12100581; 13262982; 14543914; 16012396; 17530324; 18890446; 20271441; 21938592; 23528547; 24794408; 25863033];

% Validation data (days 55-173, not used in optimization, with vaccine)
C_validation = [26720283; 27390855; 27864340; 28330091; 28750830; 29134830; 29514381; 29932199; 30382159; 30838109; 31305326; 31711421; 32049828; 32342152; 32575502; 32757111; 32904922; 33004034; 33103599; 33184226; 33271863; 33368900; 33655410; 33877989; 34237016; 34784102; 35527158; 36413622; 37433588; 38528478; 39672729; 40695299; 41732753; 42585047; 43340735; 44022719; 44604956; 45113499; 45639685; 46149896; 46707076; 47383952; 47976346; 48727363; 49562332; 50461441; 51878860; 54590898; 59273795; 64924728; 69967616; 73756878; 75945588; 77167700; 77875856; 78334128; 78666637; 78900969; 79109709; 79303178; 79497642; 79698834; 79946773; 80247494; 80626043; 81091701; 81696656; 82431151; 83171871; 83847498; 84588542; 85260798; 85970435; 86744207; 87487976; 88376205; 89284900; 90232883; 91020934; 91722552; 92367477; 92977806; 93558169; 94007678; 94414579; 94783304; 95104538; 95385122; 95636402; 95920136; 96175292; 96442596; 96725493; 97000477; 97300648; 97603247; 98065470; 98511484; 99019493; 99411696; 99883410; 100331502; 100639527; 100934394; 101219796; 101498024; 101752396; 102019564; 102247392; 102417985; 102544598; 102697566; 102873924; 102977396; 103081453; 103179920; 103266404; 103343569; 103436829];

N = 3.3e8;
Y0 = [N-7; 7; 0];

% A_guess: [Beta0, Gamma, Decay, Beta_Min]
A_guess = [1.2, 0.1, 0.05, 0.08]; 

% SIR model with baseline beta
dY = @(t, Y, b0, g, d, b_min) [
    -( ((b0-b_min)*exp(-d*t) + b_min) * Y(1) * Y(2)) / N; 
     ( ((b0-b_min)*exp(-d*t) + b_min) * Y(1) * Y(2)) / N - g * Y(2); 
    g * Y(2)
];

% Optimization (using only the first 54 points)
A_final = fminsearch(@(A) LSERR(dY, 54, Y0, C_train, A(1), A(2), A(3), A(4)), A_guess);

% Final simulation (projection to 173 days)
t_span = 1:173;
[t_out, Y_out] = ode45(@(t, Y) dY(t, Y, A_final(1), A_final(2), A_final(3), A_final(4)), t_span, Y0);
Sim_Total = Y_out(:, 2) + Y_out(:, 3);

% Plotting results
figure('Position', [100 100 900 600]);

% Training data (red)
plot(1:54, C_train, 'ro', 'MarkerSize', 4, 'DisplayName', 'Training Data (Days 1-54)'); 
hold on;

% Validation data (green)
plot(55:173, C_validation, 'go', 'MarkerSize', 4, 'DisplayName', 'Actual Data (With Vaccine)');

% Model prediction (blue)
plot(t_out, Sim_Total, 'b-', 'LineWidth', 2.5, 'DisplayName', 'SIR Model (Without Vaccine)');

title('COVID-19: Without Vaccine vs. With  Vaccine');
xlabel('Weeks'); ylabel('Total Cases');
legend('Location', 'NorthWest'); 
grid on;

% Functions:
function output = LSERR(dY, t, Y0, C, b0, g, d, b_min)
    [~, Y] = ode45(@(t, Y) dY(t, Y, b0, g, d, b_min), 1:t, Y0);
    Predicted_Total = Y(:, 2) + Y(:, 3);
    % Divide error by maximum C to produce percentage, less likely to break
    % program
    output = sum( ((Predicted_Total - C)/max(C)).^2);
end