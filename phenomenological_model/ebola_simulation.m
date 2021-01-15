f = readtable('cases_urb.csv');
sierra = f(5:end, :);

x = string(regexp(sierra.Var1, "to (\d+\s\w+\s\d*)", 'tokens'));
x = datetime(x, 'InputFormat', 'd MMMM yyyy');
y = str2double(sierra.SierraLeone_2);

idx = ~isnan(y);
x = x(idx); x = x(5:end);  % simply fitting to the reasearch article data 
y = y(idx); y = y(5:end); 
x = days(x-min(x));
x = x(x <= 112);
y = y(x <= 112);

% fitting

% generalized growth model dC/dt = rC^p. DO NOT CONFUSE C(t) with t
% p(1) ~ r
% p(2) ~ p
ggm = @(p, t) p(1) .* ((1 - p(2)) .* p(1) .* t + 2) .^ (p(2) / (1 - p(2)));

options = optimoptions('lsqcurvefit','Algorithm','levenberg-marquardt');
initial_params = [0.1, 0.1];
x1 = x; y1 = y;
fit_params = lsqcurvefit(ggm, initial_params, x1, y1, [], [], options);

% graphing
ax0 = subplot(3,1,1);
hold on
plot(x, y, 'bo', 'DisplayName','Empirical');
plot(x1, ggm(fit_params, x1), 'r', 'DisplayName','Fit');
ttl = sprintf('Generalized growth model fitting, p = %.2f, r = %.2f', fit_params(2), fit_params(1));
title(ttl);
xlabel('Days');
ylabel('Case incidence');
legend('Location','southeast');
hold off


N = 200;
sampling_params = zeros(N, 2);
for i=1:N
    ynew = poissrnd(y1);
    sampling_params(i, :) = lsqcurvefit(ggm, initial_params, x1, ynew, [], [], options);
end


[mu_p, ci_p] = confidence_interval(sampling_params(:, 2));
[mu_r, ci_r] = confidence_interval(sampling_params(:, 1));

ax1 = subplot(3,1,2); 
histogram(sampling_params(:, 2));
ttl1 = sprintf('Simulation p = %.2f (95%% CI: %.2f; %.2f)', mu_p, ci_p(1), ci_p(2));
title(ttl1);
xlabel('Deceleration of growth, 0 \leq p \leq 1');
ylabel('Frequency');

ax2 = subplot(3,1,3);
histogram(sampling_params(:, 1));
ttl2 = sprintf('Simulation r = %.2f (95%% CI: %.2f; %.2f)', mu_r, ci_r(1), ci_r(2));
title(ttl2);
xlabel('Growth rate, r');
ylabel('Frequency');


function [mu, ci] = confidence_interval(x)
%    if length(x) > 30
%        error("The sample length > 30. Check if T distribution is still applicable");
%    end
    mu = mean(x);
    se = std(x);% /sqrt(length(p_distr));
    ts = tinv([0.025  0.975], length(x) - 1);      % T-Score
    ci = mu + ts * se;
end