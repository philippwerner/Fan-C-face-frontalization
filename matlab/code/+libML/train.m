% This file is part of https://github.com/philippwerner/Fan-C-face-frontalization
% Authors: Philipp Werner and Frerk Saxen
% License: BSD 2-Clause License (see LICENSE file in root directory)

function [ model ] = train( data, ml_param )
%   Train model

    % Use linear model if nothing was set
    if ~isfield(ml_param, 'type')
        ml_param.type = 'lm';
    end
    
    if ~isfield(ml_param, 'num_cpu_cores')
        ml_param.num_cpu_cores = 1;
    end
    
    % Create model structure and set predictor index
    model = struct();
    model.predictor_idx = data.predictor_idx;

    % Train model for each selected predictor
    if length(data.predictor_idx) > 1
        % Multiple predictors are selected. We might need to train a model
        % for each predictor seperately. But this depends on each machine
        % learning method
        switch ml_param.type
            case {'lm','RFc','RFr','SVM','SVR','SVMb','SVMm','EasyEnsemble','Ensemble'}
                model.wrap_predictor = true;
            otherwise
                model.wrap_predictor = false;
        end

        if model.wrap_predictor
            
            % Prepare data for training
            n_models = length(data.predictor_idx);
            predictor_model = cell(n_models,1);

            predictor_ml_param = ml_param;
            predictor_ml_param.this_is_a_wraped_predictor = 1; 
                
            
            if 1
                % The usual way
                t0 = tic;
                
                predictor_data = data;
                
                for i = 1 : n_models
                    
                    predictor_data.predictor_idx = data.predictor_idx(i);
                    
                    % train
                    predictor_model{i} = libML.train(predictor_data, predictor_ml_param);

                    % Output progress
                    dt = toc(t0) / i;
                    if dt * n_models > 10 && (dt > 5 || mod(i, ceil(5/dt)) == 0)
                        fprintf('libML.train: model %i/%i, %.0f/%.0f seconds\n', i, n_models, i*dt, n_models*dt);
                    end
                end
            end
            
            model.predictor_model = predictor_model;
            model.ml_param = ml_param;
            return;
        end
    end
    
    % -> Only one predictor
    
    % Run parameter grid search if desired
    if isfield(ml_param, 'param_search')
        ml_param = libML.param_grid_search(data, ml_param);
    end
    
    % Subsample data if desired
    if isfield(ml_param, 'num_samples')
        n_curr = length(data.sample_idx);
        if ml_param.num_samples < n_curr
            data.sample_idx = randperm(n_curr, ml_param.num_samples)';
        end
    end
    
    % Remove samples with NaN predictor value from training set
    non_nan_idx = ~isnan(data.y(data.sample_idx, data.predictor_idx));
    data.sample_idx = data.sample_idx(non_nan_idx);
    
    % Train model for one predictor
    switch ml_param.type
       case {'SVM', 'SVR', 'SVMb', 'SVMm'}
            % Dataset will be checked in libSvm.train already
            model.svm = libSvm.train(data, ml_param.svm_param);
       otherwise
            error(strcat('Training type: ', ml_param.type, ' not supported.'));
    end
    
    % Safe ml_param on top level (not wraped predictors)
    if ~isfield(ml_param, 'this_is_a_wraped_predictor')
        model.ml_param = ml_param;
    end
    
end
