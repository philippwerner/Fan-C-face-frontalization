% This file is part of https://github.com/philippwerner/Fan-C-face-frontalization
% Authors: Philipp Werner and Frerk Saxen
% License: BSD 2-Clause License (see LICENSE file in root directory)

% This script trains a FaN-C normalization model using the SyLaFaN dataset (for use with other data).


%% Configuration

% Database root directory (edit if needed!)
db_dir = '../DB_SyLaFaN/';

% Landmarks to use (select option)
db_landmarks = 'dlib2';         % FaNC68 (including facial contour landmarks)
%db_landmarks = 'dlib2_inner';   % FaNC51 (excluding facial contour landmarks)

% Number of training samples to randomly select (pretrained models were built with 30k)
num_samples = 1000;

% Name of the output model file
output_model_fn = sprintf('models/new_model_%s_%d.mat', db_landmarks, num_samples);



%% Add main matlab code directory to path
addpath('code');

%% Load database
db_fn_prefix = 'sylafan';
util_load_db;

%% Load or create mirrored db_norm database (db_raw is not needed)
mirrored_db_fn = strcat(db_fn_prefix, '_', db_landmarks, '_db_norm_mirror.mat');
try
    load(mirrored_db_fn, 'db_norm_mirror');
catch
    fprintf('Creating mirrored db_norm database. This may take some minutes ...\n');
    
    %% Get the symmetry point for each landmark point
    first_lm_c2 = horzcat(db_norm.lm(1,1:2:end)', db_norm.lm(1,2:2:end)');
    x_mirrored = -first_lm_c2(:,1);
    [~, lm_symm_corresp] = util_pdist2_fast(first_lm_c2, horzcat(x_mirrored, first_lm_c2(:,2)), 'euclidean', 'Smallest', 1); 

    db_norm_mirror = db_norm;

    %% negate yaw
    db_norm_mirror.pose(:,2) = -db_norm_mirror.pose(:,2);

    %% negate x coords
    db_norm_mirror.lm(:,1:2:end) = -db_norm_mirror.lm(:,1:2:end);
    db_norm_mirror.pt(:,1:2:end) = -db_norm_mirror.pt(:,1:2:end);

    %% mirror visibility
    db_norm_mirror.pt_vis = db_norm_mirror.pt_vis(:,db_norm.mesh_symm_corresp);

    for i = 1:size(db_norm.lm,1)
        lm_old = db_norm.lm(i,:);
        pt_old = db_norm.pt(i,:);
        vis_old = db_norm.pt_vis(i,:);

        lm_new = db_norm_mirror.lm(i,:);
        pt_new = db_norm_mirror.pt(i,:);
        vis_new = db_norm_mirror.pt_vis(i,:);

        %% swap points with symmetric points
        lm_new = horzcat(lm_new(1:2:end)', lm_new(2:2:end)');
        lm_new = lm_new(lm_symm_corresp, :);
        lm_new = reshape(lm_new', 1, []);

        pt_new = horzcat(pt_new(1:2:end)', pt_new(2:2:end)');
        pt_new = pt_new(db_norm.mesh_symm_corresp, :);
        pt_new = reshape(pt_new', 1, []);


        %% pre-alignment
        tform = tform_fit_prealign(lm_new);
        lm_new = tform_forward(lm_new, tform);
        pt_new = tform_forward(pt_new, tform);

        db_norm_mirror.lm(i,:) = lm_new;
        db_norm_mirror.pt(i,:) = pt_new;

        %% plot (insert breakpoint to see visualization)
        if 0
            plot(lm_new(1:2:end), lm_new(2:2:end), 'rx-');
            hold on;
            pt_new_x = pt_new(1:2:end);
            pt_new_y = pt_new(2:2:end);
            plot(pt_new_x, pt_new_y, 'r.');
            plot(pt_new_x(~vis_new), pt_new_y(~vis_new), 'ro');

            plot(lm_old(1:2:end), lm_old(2:2:end), 'bx-');
            pt_old_x = pt_old(1:2:end);
            pt_old_y = pt_old(2:2:end);
            plot(pt_old_x, pt_old_y, 'b.');
            plot(pt_old_x(~vis_old), pt_old_y(~vis_old), 'bo');
            hold off; axis ij;
        end
    end
    
    save(mirrored_db_fn, 'db_norm_mirror');
end

%% only use mirrored expression of asymetric expressions (and original images)
eid = db_norm_mirror.expression_id;
idx = (eid == 12 | eid == 13 | eid == 17);
db_norm_both = db_norm;
db_norm_both.sample_id = vertcat(db_norm.sample_id, db_norm_mirror.sample_id(idx,:) + max(db_norm.sample_id));
db_norm_both.dist = vertcat(db_norm.dist, db_norm_mirror.dist(idx,:));
db_norm_both.subject_id = vertcat(db_norm.subject_id, db_norm_mirror.subject_id(idx,:));
db_norm_both.expression_id = vertcat(db_norm.expression_id, db_norm_mirror.expression_id(idx,:) + max(db_norm.expression_id));  % trick: mirrored are additional expression
db_norm_both.seq_id = vertcat(db_norm.seq_id, db_norm_mirror.seq_id(idx,:));
db_norm_both.frame_id = vertcat(db_norm.frame_id, db_norm_mirror.frame_id(idx,:));
db_norm_both.pose = vertcat(db_norm.pose, db_norm_mirror.pose(idx,:));
db_norm_both.lm = vertcat(db_norm.lm, db_norm_mirror.lm(idx,:));
db_norm_both.pt = vertcat(db_norm.pt, db_norm_mirror.pt(idx,:));
db_norm_both.pt_vis = vertcat(db_norm.pt_vis, db_norm_mirror.pt_vis(idx,:));
db_norm_both.corresponding_frontal_idx = vertcat(db_norm.corresponding_frontal_idx, db_norm_mirror.corresponding_frontal_idx(idx,:) + size(db_norm.sample_id,1));
db_norm_both.xm2_feat = [];
    



%% Model training config: corresp. point coordinate prediction
pt_svr_param = struct();
pt_svr_param.type = 'SVR';
pt_svr_param.library = 'liblinear';
pt_svr_param.epsilon = '0.005';
pt_svr_param.kernel = 'linear';
pt_svr_param.C = 2 ^ -2;

pt_ml_param = struct();
pt_ml_param.type = 'SVM';
pt_ml_param.svm_param = pt_svr_param;
pt_ml_param.num_cpu_cores = 6;

%% Model training config: corresp. point visibility prediction
vis_svm_param = struct();
vis_svm_param.type = 'SVMm';
vis_svm_param.library = 'liblinear';
vis_svm_param.predict_fast = true;
vis_svm_param.C = 1;

vis_ml_param = pt_ml_param;
vis_ml_param.redistribute_param = struct(...
    'type', 'min', ...
    'factor', 1, ...
    'downsampling_type', 'random', ...
    'upsampling_type', 'none', ...
    'visualize', false, 'figure', 2);
vis_ml_param.num_samples = num_samples;
vis_ml_param.svm_param = vis_svm_param;
vis_ml_param.num_cpu_cores = 6;

%% Model training config: more parameters
param = struct;
param.pt_ml_param = pt_ml_param;
param.vis_ml_param = vis_ml_param;
param.train_sample_count = num_samples; % for pt_model only


%% Train and save model
rng('default'); % Set seed for RNG
t0 = tic;
regfunc_fanc_train( db_norm_both, param, output_model_fn);
total_runtime = toc(t0)



