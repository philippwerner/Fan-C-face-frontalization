% This file is part of https://github.com/philippwerner/Fan-C-face-frontalization
% Authors: Philipp Werner and Frerk Saxen
% License: BSD 2-Clause License (see LICENSE file in root directory)

function [ y, p ] = predict( data, model, ml_param )
% Predict response from features
%
% [ y, p ] = predict( data, model )
%   model: model returned by ml.train
%   dataset: Struct containing 
%       .x = Entire feature dataset (one row per sample)
%       .sample_idx = Index of samples to predict.
%
%   y: Output responses (binary or regression)
%
%   p: Ensemble scores or SVM prediction matrix (or empty)
%
% [ y, p ] = predict( data, model, ml_param )
%
%   ml_param: passing parameters for wraped models.
%

    % Check input
    if ~isstruct(model)
        error('Input model must be created by +libML.train.');
    end

    % If ml_param is not given, use from model
    if nargin < 3
        ml_param = model.ml_param;
    end

    % Check if dataset is legit
    if ~isfield(ml_param, 'skip_data_check') || ~ml_param.skip_data_check
        libDataset.util_check_dataset(data);
        % we can safely skip check in child calls
        ml_param.skip_data_check = 1;
    end
    
    % Most classifiers don't provide scores :(
    p = [];

    % Are there several predictors and we got to wrap them up?
    % Unfortunately, we can not provide the scores because they are
    % inconsistent from one classifier to another.
    num_predictors = length(model.predictor_idx);
    if num_predictors > 1 && model.wrap_predictor
        y = zeros(size(data.sample_idx,1), num_predictors);
        p = y;
        ps = 1;
        for i = 1 : num_predictors
            data.predictor_idx = model.predictor_idx(i);
            [y(:,i), pi] = libML.predict(data, model.predictor_model{i}, ml_param);
            if nargout > 1
                if i == 1 && size(pi, 2) ~= 1 % Adjust p if the probs don't exist or have more dimensions than just one.
                    ps = size(pi, 2);
                    p = zeros(size(data.sample_idx,1), num_predictors * ps);
                end
                p(:, (i-1)*ps+1 : i*ps) = pi;
            end
        end
        return;
    end



    % Apply primary model
    switch ml_param.type
        case {'SVM', 'SVMm', 'SVR', 'SVMb'}
            [y, p] = libSvm.predict(data, model.svm, ml_param.svm_param);
        otherwise
            % Use matlab buildin prediction
            [y, p] = predict(model.ml, data.x(data.sample_idx, :));
            if iscell(y)
                y = str2double(y);
            end
    end
    
end
