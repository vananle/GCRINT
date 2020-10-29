classdef Util
    methods
        function [rse, mae, mape, mse, rmse] = get_performance(self, X, Xhat, W)
            eps = 1e-3;

            Wr = ~W;
            X_w = X .* Wr;
            Xhat_w = Xhat .* Wr;
            err = Xhat_w(:) - X_w(:);
            rse = sqrt(sum(err .^ 2) / sum(X_w(:) .^ 2));
            mae = mean(abs(err));
            
            X_w(find(X_w < eps)) = eps;
            mape = mean(abs(err ./ X_w(:)));
            mse = mean(err .^ 2);
            rmse = sqrt(mse);
        end
    end
end
