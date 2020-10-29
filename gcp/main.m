function run_experiment(dataset, type)
    modes = {'train', 'val', 'test'};
    srs = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    repeat = 1;

    % run all scenarios
    results = [];
    % prepare parameter
    args.dataset = dataset;
    args.type = type;
    for i3 = 1:length(srs)
        for i4 = 1:repeat
            for i5 = 1:length(modes)
                args.sr = srs{i3};
                args.seed = i4;
                args.mode = modes{i5};
                args.num_iter = 30;
                [rse, mae, mape, mse, rmse, t] = gcp(args);
                % save result
                path = sprintf('result/%s.csv', args.dataset);
                fp = fopen(path, 'a');
                fprintf(fp, '%s,%s,%s,%0.1f,%i,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f\n', ...
                        args.dataset, args.type, args.mode, args.sr, args.seed, ...
                        rse, mae, mape, mse, rmse, t);
                fclose(fp);
            end
        end
    end
end
