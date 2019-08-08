% This file is part of https://github.com/philippwerner/Fan-C-face-frontalization
% Authors: Philipp Werner and Frerk Saxen
% License: BSD 2-Clause License (see LICENSE file in root directory)

% This script selects a random sample of the LFW dataset, applies the FaN-C method to frontalize the sample,
% and visualizes the result. This is a real cross-database test, because LFW has not been used at all during
% the development and training of the Fan-C method and model.


%% Configuration

% Number of random samples to show
n = 20;

% Seconds to wait before showing the next image
wait_sec = 1;

% Database root directory (edit if needed!)
image_root_dir = '../lfw/';

% Landmarks to use (select option)
db_landmarks = 'dlib2';         % FaNC68 (including facial contour landmarks)
%db_landmarks = 'dlib2_inner';   % FaNC51 (excluding facial contour landmarks)

% Whether to keep the background
keep_background = false;

% Resolution of output image
output_resolution = [200 200];

% whether to show visualizations
visualize = true;


%% Add main matlab code directory to path
addpath('code');

%% Load LFW landmarks (localized our custom dlib model, see README.md)
lfw_landmarks_file = 'data/lfw_dlib2_lm_raw.mat';
load(lfw_landmarks_file);

%% Load FaNC model
corresp_mode = 'PoseIdentityNorm';
load(['models/m170306_', db_landmarks, '_xm2.mat']);
model.to_size = output_resolution;
model.show_background = keep_background;
%model.gpu_warper = libFastWarp.FastWarp();

%% Sample selection (default: n samples of random order)
for i = randperm(dblm.sample_cnt, n);

    %% Load image and landmarks
    in_fn = fullfile(image_root_dir, strcat(dblm.rel_filenames{i}, '.jpg'));
    in_img = imread(in_fn);
    if strcmp(db_landmarks, 'dlib2_inner')
        lm = dblm.data(i,2*17+1:end)+1;
    else
        lm = dblm.data(i,:)+1;
    end

    %% Apply Fan-C face normalization
    [out_img, res] = regfunc_fanc_do_normalization(in_img, lm, model);

    %% Visualize
    if visualize
        figure(1);

        subplot(121);
        imshow(in_img); hold on; plot(lm(1:2:end), lm(2:2:end), 'g.'); hold off;
        title('Input image with landmarks');

        subplot(122); 
        imshow(out_img);
        title('Result of Fan-C frontalization');
    end
    
    %% Pause for wait_sec seconds (or press button to switch faster)
    pause(wait_sec);

end
