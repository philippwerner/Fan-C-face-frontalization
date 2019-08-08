% This file is part of https://github.com/philippwerner/Fan-C-face-frontalization
% Authors: Philipp Werner and Frerk Saxen
% License: BSD 2-Clause License (see LICENSE file in root directory)

function regfunc_fanc_train( db_norm, param, model_fn )

corresp_t_idx = db_norm.get_corresponding_frontal_idx('PoseIdentityNorm', true(size(db_norm.lm,1), 1));
[gt_pt_vis, gt_pt_vis2] = db_norm.get_visibility_masks();

if ~isfield(param, 'feature_mode')
    param.feature_mode = 'xm2';
end

if ~isfield(param, 'predict_pc_scores')
    param.predict_pc_scores = false;
end

%% Create feature/sample matrix (s: source, t: target)
sn_gt_pt = db_norm.pt;
tn_gt_pt = db_norm.pt(corresp_t_idx,:);
lm_feat_s = db_norm.lm;
lm_feat_t = db_norm.lm(corresp_t_idx,:);

if ~param.predict_pc_scores
    % no need to predict invisible mesh points in source domain -> exclude
    % from training and test data
    sn_gt_pt(~gt_pt_vis2) = NaN;
end

if param.predict_pc_scores
    % apply pca on predictors to get new predictors
    params_sn = struct('feature_pca', true, 'feature_pca_var_to_keep', 0.99);
    [ sn_gt_pt, params_sn ] = model_prepare_features( sn_gt_pt, params_sn );
    params_tn = struct('feature_pca', true, 'feature_pca_var_to_keep', 0.99);
    [ tn_gt_pt, params_tn ] = model_prepare_features( tn_gt_pt, params_tn );
end


if strcmp(param.feature_mode, 'xm2') && ~isempty(db_norm.xm2_feat)
    % load features directly if available
    X = db_norm.xm2_feat;
    Y = horzcat(sn_gt_pt, tn_gt_pt);
else
    % prepare feature extraction (calculate pca if necessary)
    param.feature_mode = regfunc_fanc_train_feature_extract(param.feature_mode, lm_feat_s);

    % extract features, arrange labels
    [X, Y] = regfunc_fanc_create_samples(param.feature_mode, lm_feat_s, sn_gt_pt, lm_feat_t, tn_gt_pt);
end


%% Create training dataset
dataset = libDataset.create_dataset(X, Y, db_norm.subject_id);
dataset.sample_id = db_norm.sample_id;

% coordinate pred.
pt_train_dataset = dataset;
pt_train_dataset = libDataset.normalize(pt_train_dataset);

% visibility pred.
num_predictors = size(db_norm.pt_vis, 2);
vis_train_dataset = pt_train_dataset;
vis_train_dataset.y = double(db_norm.pt_vis);
vis_train_dataset.predictor_type = zeros(num_predictors, 1);
vis_train_dataset.predictor_idx = (1 : num_predictors)';

if isfield(param, 'train_sample_count')
    % reduce sample count for training
    n_cur = length(pt_train_dataset.sample_idx);
    n_want = param.train_sample_count;
    if n_want < n_cur
        idx = randperm(n_cur, n_want);
        % normally do not apply to the visibility data, because rebalancing is
        % applied later
        pt_train_dataset.sample_idx = pt_train_dataset.sample_idx(idx);
        if isfield(param, 'train_sample_count_also_for_vis') && param.train_sample_count_also_for_vis
            vis_train_dataset.sample_idx = vis_train_dataset.sample_idx(idx);
        end
    end
    clear n_cur n_want idx;
end

clear dataset;



%% Training pt model
fprintf('Training point coordinate models\n');
pt_model = libML.train(pt_train_dataset, param.pt_ml_param);


%% Training vis model
fprintf('Training point visibility models\n');
vis_model = libML.train(vis_train_dataset, param.vis_ml_param);


%% Save models
left_pt_idx = db_norm.pt(1,1:2:end) < 0.05;     % include middle points
right_pt_idx = db_norm.pt(1,1:2:end) > -0.05;

model = struct( ...
    'predict_pc_scores', param.predict_pc_scores, ...
    'feature_mode', param.feature_mode, ...
    'feature_norm', pt_train_dataset.norm_values, ...
    'pt_model', pt_model, ...
    'vis_model', vis_model, ...
    'train_img_idx', pt_train_dataset.sample_idx, ...
    'left_pt_idx', left_pt_idx, ...
    'right_pt_idx', right_pt_idx, ...
    'warp_pt_sym_corresp', db_norm.mesh_symm_corresp, ...
    'warp_triangles', db_norm.triangles);
if param.predict_pc_scores
    model.pca_sn = params_sn;
    model.pca_tn = params_tn;
end
save(model_fn, 'model', '-v7.3');


fprintf('Training done... saved models\n');

end

