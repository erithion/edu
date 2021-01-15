% Calculates the expected value of iterations for the last occurence of a
% major number (>N/2) based on both empirical simulation and analytic
% calculation. See empirical_Ev and analytic_Ev
clear all; 

population_N = 200;
simulation_N = 100000;
major_N = population_N / 2;

res = zeros(simulation_N, 1);
p_first = major_N / population_N;
p_rest = (1 - p_first) / (population_N - 1);
weights = [p_first; p_rest * ones(population_N - 1, 1)];
for i = 1:simulation_N
    population = randperm(population_N);
    sample = randsample(population, population_N, true, weights);
    idx = find(sample == population(1), 1, 'last');
    res(i) = idx;
end

% hold on
%iters = unique(res);
% x = [[1:population_N - length(iters)].' ; iters];
% h = histogram(res, x, 'Normalization', 'probability');
% distribution
% d = histcounts(res, iters, 'Normalization', 'probability');
[emp_counts, emp_iters] = groupcounts(res);
emp_counts = emp_counts / simulation_N;
subplot(2,1,1);
bar(emp_iters, emp_counts);
% expected value
empirical_Ev = 0;
n = length(emp_iters);
for i = 1:n
    empirical_Ev = empirical_Ev + emp_iters(i) * emp_counts(i); 
end


N = population_N;
K = major_N;
T = N - K;
analytic_Ev = 0;
prob = zeros(N, 1);
mul = K/N;
d1 = T;
d2 = N - 1;
for i = N:-1:K 
    prob(i) = mul;
    analytic_Ev = analytic_Ev + i * prob(i);
    mul = mul * d1 / d2;
    d1 = d1 - 1;
    d2 = d2 - 1;
end
idx = prob ~= 0;
ana_iters = find(idx);
ana_counts = prob(idx);
subplot(2,1,2);
bar(ana_iters, ana_counts);

% too big factorials
%{ 
N_ = population_N;
K_ = major_N;
T_ = N_ - K_;
S_ = factorial(N_);
analytic_Ev_ = 0;
prob_ = zeros(N_, 1);
for i = 1:N_
    prob_(i) = npermk(T_, N_ - i) * K_ * npermk(i - 1, i - 1) / S_;
    analytic_Ev_ = analytic_Ev_ + i * prob_(i);
end
idx_ = prob_ ~= 0;
ana_iters_ = find(idx_);
ana_counts_ = prob_(idx_);
subplot(3,1,3);
bar(ana_iters_, ana_counts_);

function p = npermk(n, k)
    if (k > n) 
        p = 0;
    else
        p = factorial(n) / factorial(n - k);
    end
end
%}


