% This file is part of https://github.com/philippwerner/Fan-C-face-frontalization
% Authors: Philipp Werner and Frerk Saxen
% License: BSD 2-Clause License (see LICENSE file in root directory)

function ml_param = ml_param()
% Mashine learning parameters for ml.train().
%
% ml_param
%   .type =
%       'lm':   Fit linear model using lmfit (default)
%       'RFc':  Random Forest classification
%       'RFr':  Random Forest regression
%       'SVM', 'SVR':   Support Vector Machine or Regression. Please
%                       specify in ml_param.svm_param.
%       'EasyEnsemble': Easy Ensemble classification.
%       'Ensemble'    : Splits the dataset into several chunks and trains a
%                       new model for each set. An aggregation model
%                       combines all trained model to provide a single
%                       output.
%   
%   .num_cpu_cores =
%       Number of cpu cores to use for training. Only used if multiple
%       predictors available for training. (default 1)
%
%   .svm_param =
%       Structure of svm parameter used when ml_param.type = 'SVM' or 'SVR'.
%       See svm.create_svm_param.m for details.
%
%   .ee_param =
%       Easy Ensemble parameter, used when ml_param.type = 'EasyEnsemble'.
%       See EasyEnsemble.create_EasyEnsemble_param.m for details.
%
%   .ensemble_param =
%       Nested ml_param to train an ensemble of SVM, SVR, or RFc or
%       whatsoever. Used when ml_param.type = 'Ensemble'. See
%       ml.ml_param.m for details ;) 
%
%   .ensemble_num_models =
%       Number of models in the ensemble. Used, when ml_param.type =
%       'Ensemble'. (default 4)
%
%   .num_samples =
%       Number of samples used for training. If
%       ml_param.type ~= 'Ensemble', the default is to use all samples.
%       With ml_param.type = 'Ensemble', each ensemble model will get
%       independently sampled data for training. By default its set to the
%       number of samples in the dataset times 2 / ensemble_num_models.
%       Thus, all sampled subsets together contain twice the number of
%       samples compared to the original dataset.
%
%   .redistribute_param =
%       Redistribution parameter to change the skew of the dataset. Please
%       see libDataset.redistribute_param.m for parameter details.

ml_param = struct;

end