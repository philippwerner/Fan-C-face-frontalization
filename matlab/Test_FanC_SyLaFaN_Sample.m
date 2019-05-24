% This file is part of https://github.com/philippwerner/Fan-C-face-frontalization
% Authors: Philipp Werner and Frerk Saxen
% License: BSD 2-Clause License (see LICENSE file in root directory)

% This script selects a random sample of the SyLaFaN dataset, applies the FaN-C method to frontalize the sample,
% and visualizes some details.
%
% Note that the sample may be part of the training set used to create the normalization model (in contrast to the
% examples in the paper, which were selected from the test set of a cross-validation (without subject overlap).


%% Configuration

% Database root directory (edit if needed!)
db_dir = '../DB_SyLaFaN/';

% Landmarks to use
db_landmarks = 'dlib2';         % FaNC68 (including facial contour landmarks)
%db_landmarks = 'dlib2_inner';   % FaNC51 (excluding facial contour landmarks)

% Resolution of output image
output_resolution = [400 360];

% whether to show visualizations
visualize = true;

% whether to plot ground truth
plot_gt = true;


%% Add main matlab code directory to path
addpath('code');

%% Load database
db_fn_prefix = 'sylafan';
util_load_db;

%% Load FaNC model
corresp_mode = 'PoseIdentityNorm';
load(['models/m170306_', db_landmarks, '_xm2.mat']);
model.to_size = output_resolution;
%model.gpu_warper = libFastWarp.FastWarp();



%% Sample selection (default: random)
i = round(rand * length(db_raw.sample_id));
%i = 100;

%% Notation:
% sr_*     source image raw image coordinates (in database)
% sn_*     source image normalized coordinates (for machine learning)
% tr_*     target image raw image coordinates (in database)
% tn_*     target image normalized coordinates (for machine learning)
% to_*     target output image coordinates (for warped image)
% *_in_*   input
% *_pred_* predicted
% *_gt_*   ground truth
% *_lm     detected landmark points
% *_pt     correspondence points

%% Load input
sr_in_fn = db_raw.get_image_filename(i);
%sr_in_img = rgb2gray(imread(sr_in_fn));
sr_in_img = imread(sr_in_fn);
sr_in_lm = db_raw.lm(i,:);

fprintf('%s\n', sr_in_fn);



%% Visualize input
if visualize
    figure(1);
    subplot(2,2,1);
    imagesc(sr_in_img);
    colormap('gray'); axis('ij'); axis('image');
    title('query input: source image, raw coord');
    hold on; plot(sr_in_lm(:,1:2:end), sr_in_lm(:,2:2:end), 'b.'); hold off;
end

%% Run Fan-C face normalization algorithm
[ to_pred_img, interm_results ] = ...
    regfunc_fanc_do_normalization( sr_in_img, sr_in_lm, model);


%% Visualize prealignment
sr_sn_tform = interm_results.sr_sn_tform;
sn_in_lm = interm_results.sn_in_lm;

if visualize
    figure(1);
    sr_sn_tform_img = sr_sn_tform;
    sr_sn_tform_img.T = sr_sn_tform_img.T(:,1:2) * 100;
    sn_in_img = imwarp(sr_in_img, sr_sn_tform_img, 'OutputView', imref2d([300 300], [-150 150], [-150 150]));

    subplot(2,2,2);
    imagesc([-1.5 1.5], [-1.5 1.5], sn_in_img);
    colormap('gray'); axis('ij'); axis('image');
    title('source image, norm coord');
    hold on; plot(sn_in_lm(:,1:2:end), sn_in_lm(:,2:2:end), 'gx'); hold off;
    hold on; plot([-0.5 0.5], [-0.5 -0.5], 'yo'); hold off;
    
    subplot(2,2,3);
    imagesc(to_pred_img); axis image;
    title('output image');
end

%% Visualize source and target DB images with ground truth points and landmarks
if plot_gt
    sr_gt_pt = db_raw.pt(i,:);
    
    sn_gt_pt = db_norm.pt(i,:);
    
    sn_gt_vis = logical(db_raw.pt_vis(i,:));
    
    corr_frontal_idx = db_norm.get_corresponding_frontal_idx(corresp_mode);
    tn_gt_pt = double(db_norm.pt(corr_frontal_idx(i),:));
    
    figure(2);

    % Source
    subplot(2,1,1);
    imagesc(sr_in_img); colormap('gray'); axis('ij'); axis('image'); title('source image, db content');
    hold on;
    c2_gt_pt = horzcat(sr_gt_pt(:,1:2:end)', sr_gt_pt(:,2:2:end)');
    plot(c2_gt_pt(sn_gt_vis,1), c2_gt_pt(sn_gt_vis,2), 'bx');
    plot(c2_gt_pt(~sn_gt_vis,1), c2_gt_pt(~sn_gt_vis,2), 'rx');
    plot(sr_in_lm(:,1:2:end), sr_in_lm(:,2:2:end), 'gx');
    hold off;

    % Target
    tr_in_fn = db_raw.get_image_filename(corr_frontal_idx(i));
    %tr_in_img = rgb2gray(imread(tr_in_fn));
    tr_in_img = imread(tr_in_fn);
    tr_in_lm = db_raw.lm(corr_frontal_idx(i),:);
    tr_gt_pt = db_raw.pt(corr_frontal_idx(i),:);
    subplot(2,1,2);
    imagesc(tr_in_img); colormap('gray'); axis('ij'); axis('image'); title('target image, db content');
    hold on;
    plot(tr_gt_pt(1:2:end), tr_gt_pt(2:2:end), 'bx');
    plot(tr_in_lm(:,1:2:end), tr_in_lm(:,2:2:end), 'gx');
    hold off;
end



