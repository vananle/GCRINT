function [rse, mae, mape, mse, rmse, t] = gcp(args)
    % load data
    path = sprintf('../../data/data/%s.mat', args.dataset);
    load(path);
    path = sprintf('../../data/mask/%s/%s/%0.1f_%i.mat', ...
                    args.dataset, args.type, args.sr, args.seed);
    load(path);

    % 3D reshape: T, O, D
    [T, N] = size(X);
    D = sqrt(N);
    X = reshape(X, [T, D, D]);
    W = reshape(W, [T, D, D]);

%    % Cut 20% last time step to test
%    T = round(T * 0.2);
%    X = X(end-T:end, :, :);
%    W = W(end-T:end, :, :);
    % Cut train/val/test to recover
    p_train = 0.7;
    p_val   = 0.1;
    p_test  = 0.2;
    num_train = round(T * p_train);
    num_val   = round(T * p_val);
    num_test  = round(T * p_test);
    switch args.mode
        case 'train'
            X = X(1:num_train, :, :);
            W = W(1:num_train, :, :);
        case 'val'
            X = X(num_train+1:num_train+num_val, :, :);
            W = W(num_train+1:num_train+num_val, :, :);
        case 'test'
            X = X(num_train+num_val+1:end, :, :);
            W = W(num_train+num_val+1:end, :, :);
    end

    % choose rank
    args.rank = 32;
    args.rank = min(300, args.rank);

    % display information
    fprintf('-----------------------------\n');
    fprintf('[+] GCP recovering experiment\n');
    fprintf('-----------------------------\n');
    fprintf('    - dataset : %s\n', args.dataset);
    fprintf('    - type    : %s\n', args.type);
    fprintf('    - mode    : %s\n', args.mode);
    fprintf('    - size    : %ix%ix%i\n', size(X));
    fprintf('    - sr      : %0.1f\n', args.sr);
    fprintf('    - seed    : %i\n', args.seed);
    fprintf('-----------------------------\n');
    fprintf('    - rank    : %i\n', args.rank);
    fprintf('    - num_iter: %i\n', args.num_iter);
    fprintf('-----------------------------\n');

    % convert to tensor
    X = tensor(double(X));
    W = tensor(double(W));

    % decompose
    tic;
    Xhat = gcp_opt(X, args.rank, ...
                    'mask', W, ...
                    'type', 'normal', ...
                    'maxiters', args.num_iter, ...
                    'printitn', 0, ...
                    'opt', 'lbfgsb');
    Xhat = tensor(Xhat);
    t = toc;

    % evaluate performance
    util = Util();
    [rse, mae, mape, mse, rmse] = util.get_performance(X, Xhat, W); 
    fprintf('rse=%0.4f mae=%0.4f mape=%0.4f mse=%0.4f rmse=%0.4f, t=%0.1f\n', rse, mae, mape, mse, rmse, t);

    % create folder to save the imputed data
    path = '../../result';
    if ~exist(path, 'dir')
        mkdir(path);
    end
    path = sprintf('../../result/%s_%s_%0.1f_%s', 'gcp', args.dataset, args.sr, args.type);
    if ~exist(path, 'dir')
        mkdir(path);
    end
    % convert data from tensor to regular matlab matrix
    X = tenmat(X, [1], [2 3]);
    X = X.data;
    W = tenmat(W, [1], [2 3]);
    W = W.data;
    X_imp = tenmat(Xhat, [1], [2 3]);
    X_imp = X_imp.data;
    % save the imputed data
    path = sprintf('%s/%s.mat', path, args.mode);
    save(path, 'X', 'W', 'X_imp');
end
