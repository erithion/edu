% simulation to calculate the expected value for the first occurence of the major element in the array
% i.e. that occurs more than N/2 times
clear all; 

population_N = 1000;
simulation_N = 10000;

res = zeros(simulation_N, 1);
for i = 1:simulation_N
    population = randperm(population_N);
    p_first = 0.5;
    p_rest = (1 - p_first) / (population_N - 1);
    sample = randsample(population, population_N, true, [p_first; p_rest * ones(population_N - 1, 1)]);
    idx = find(sample == population(1), 1, 'first');
    res(i) = idx;
end

histogram(res,'Normalization', 'probability');
% distribution
d = histcounts(res, 'Normalization', 'probability');
% expected value
Ev = 0;
for i = 1:length(d)
    Ev = Ev + i * d(i); 
end
