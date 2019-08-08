% This file is part of https://github.com/philippwerner/Fan-C-face-frontalization
% Authors: Philipp Werner and Frerk Saxen
% License: BSD 2-Clause License (see LICENSE file in root directory)

% usage: define db_dir, db_prefix, db_landmarks
if ~exist('db_dir', 'var') || ~exist('db_fn_prefix', 'var') || ~exist('db_landmarks', 'var')
    error(sprintf('Please define following variables before running util_load_db:\n\tdb_dir\n\tdb_fn_prefix\n\tdb_landmarks = string intraface, dlib or dlib_inner'));
end

db_load_fn_prefix = strcat('data/', db_fn_prefix, '_', db_landmarks, '_');
db_load_fn_raw = strcat(db_load_fn_prefix, 'db_raw.mat');
db_load_fn_norm = strcat(db_load_fn_prefix, 'db_norm.mat');

%% Create raw db file
if ~exist(db_load_fn_raw, 'file')
    fprintf('Creating %s for fast access. This may take some minutes ...\n', db_load_fn_raw);
    switch db_landmarks
        case 'intraface'
            lm = dlmread(fullfile(db_dir, 'landmarks_intraface.table'), '\t', 0, 0);
        case 'dlib'
            lm = dlmread(fullfile(db_dir, 'landmarks_dlib.table'), '\t', 0, 0);
        case 'dlib_inner'
            lm = dlmread(fullfile(db_dir, 'landmarks_dlib.table'), '\t', 0, 0);
            lm = lm(:,35:end);
        case 'dlib2'
            fn = fullfile(db_dir, 'landmarks_dlib2.table');
            if ~exist(fn, 'file')
                fprintf('Post-processing landmarks to create landmarks_dlib2.table ...\n');
                util_create_db_postprocess_lm_dlib2();
            end
            lm = dlmread(fn, '\t', 0, 0);
        case 'dlib2_inner'
            fn = fullfile(db_dir, 'landmarks_dlib2.table');
            if ~exist(fn, 'file')
                fprintf('Post-processing landmarks to create landmarks_dlib2.table ...\n');
                util_create_db_postprocess_lm_dlib2();
            end
            lm = dlmread(fn, '\t', 0, 0);
            lm = lm(:,35:end);
        otherwise
            error('The variable db_landmarks must be a string containing: intraface, dlib or dlib_inner!');
    end
    index = dlmread(fullfile(db_dir, 'index.table'), '\t', 1, 0); 
    pose = dlmread(fullfile(db_dir, 'pose.table'), '\t', 1, 0);
    pt = dlmread(fullfile(db_dir, 'corresp_pts.table'), '\t', 0, 0);
    pt_vis = dlmread(fullfile(db_dir, 'corresp_pts_visibility.table'), '\t', 0, 0);

    db_raw = FaceRegistrationDatabase(index, pose, lm, pt, pt_vis);
    db_raw.database_root_path = db_dir;
    save(db_load_fn_raw,'db_raw','-v7.3');
else
    load(db_load_fn_raw);
end

%% Create norm db file
if ~exist(db_load_fn_norm, 'file')
    % For in-plane rotation, scale and translation invariance: simple in-plane
    % transform based on eyes
    fprintf('Creating %s for fast access. This may take some minutes ...\n', db_load_fn_norm);
    db_norm = db_raw;
    M = NaN(size(db_raw.lm));
    for i = 1:length(db_raw.sample_id)
        tform = tform_fit_prealign(db_raw.lm(i,:));
        db_norm.lm(i,:) = tform_forward(db_raw.lm(i,:), tform);
        db_norm.pt(i,:) = tform_forward(db_raw.pt(i,:), tform);
        if 0
            figure(1);
            subplot(211);
            hold off; plot(db_raw.lm(i,1:2:end), db_raw.lm(i,2:2:end), 'b+');
            hold on; plot(db_raw.pt(i,1:2:end), db_raw.pt(i,2:2:end), 'r.'); hold off;
            axis('ij'); title('original');
            subplot(212);
            hold off; plot(db_norm.lm(i,1:2:end), db_norm.lm(i,2:2:end), 'b+');
            hold on; plot(db_norm.pt(i,1:2:end), db_norm.pt(i,2:2:end), 'r.'); hold off;
            hold on; plot([-0.5 0.5], [-0.5 -0.5], 'go'); hold off;
            axis('ij'); title('praligned');
            %xlim([-1 1]); ylim([-0.1 1.1]);
        end
        % prealign mouth
        tform = tform_fit_prealign_mouth(db_norm.lm(i,:));
        M(i,:) = tform_forward(db_norm.lm(i,:), tform);
    end
    db_norm.xm2_feat = horzcat(db_norm.lm, M, db_norm.lm .^ 2, M .^ 2);
    save(db_load_fn_norm,'db_norm');
else
    load(db_load_fn_norm);
end

clear db_load_fn_prefix db_load_fn_raw db_load_fn_norm lm index pose lm_dlib pt pt_vis;
