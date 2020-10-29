
datasets = {'abilene_tm_10k', 'brain_tm_10k', 'geant_tm_10k'};
types = {'uniform', 'block'};

for i1 = 1:length(datasets)
    for i2 = 1:length(types)
        dataset = datasets{i1};
        type    = types{i2};
        main(dataset, type);
    end
end

