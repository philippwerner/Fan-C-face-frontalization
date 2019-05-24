% This file is part of https://github.com/philippwerner/Fan-C-face-frontalization
% Authors: Philipp Werner and Frerk Saxen
% License: BSD 2-Clause License (see LICENSE file in root directory)

% usage: define db_dir

% Load data
if ~exist('data_origin', 'var') || ~strcmp(data_origin, 'dlib2')
    % subset of corrspondence points which should be close to landmarks (if landmarks are correct)
    pts = dlmread(fullfile(db_dir, 'corresp_pts_for_landmark_check.table'),'\t');
    pts = pts(:,2:end);
    
    % all landmarks (include facial contour)
    lm_all = dlmread(fullfile(db_dir, 'landmarks_dlib2_prelim.table'),' ',1,0);
    lm_all = lm_all(:,2:end-1);
    
    % head pose ground truth
    pose = dlmread(fullfile(db_dir, 'pose.table'),'\t',1,0);
    pose = pose(:,2:end);

    % sample index
    index = dlmread(fullfile(db_dir, 'index.table'),'\t',1,0);
    index = index(:,2:end);
    
    data_origin = 'dlib2';
end

% Only use inner landmarks (excluding facial contour for checks)
lm = lm_all(:,35:end);

% Identify samples with failed detection
detection_failed = all(lm_all' == -1);


% Calculate nearest neighbor for each landmark point
% with respect to the mesh points (only frontal samples are
% used)
frontal_samples = all(pose' == 0) & ~detection_failed;
k = dsearchn( ...
    [pts(frontal_samples,1:2:end)', pts(frontal_samples,2:2:end)'], ...
    [lm(frontal_samples,1:2:end)', lm(frontal_samples,2:2:end)']);
k = [(k .* 2 - 1)'; (k .* 2)'];

% Plot point correspondance
if 1
    figure(1);
    for t = 1%d_idx'%randperm(size(lm,1))
        plot(lm_all(t,1:2:end),lm_all(t,2:2:end),'or');  hold on;
        plot(pts(t,1:2:end),pts(t,2:2:end),'xb');
        for i = 1 : (size(lm,2)-1) / 2
            plot([pts(t,k(i * 2 - 1)), lm(t, 2*i-1)], [pts(t,k(i * 2)) lm(t, 2*i)], 'c');
        end
        hold off; axis image; axis ij;
    end
end

% Calculate average distance between correspondance points for
% each image in the dataset
d = (pts(:,k) - lm(:,:)).^2;
d = sqrt(d(:,1:2:end) + d(:,2:2:end));
%d = mean(d,2);
%d = max(d,[],2);
d = mean(d .^ 2,2); % penalize high error points

% Sort images by error
[~, d_idx] = sort(d, 'descend');

figure(11);
plot(sort(d(~detection_failed))); grid on;
xlabel('samples'); ylabel('landmark error'); title('cumulative distribution of estimated landmark errors');

% REMOVE SAMPLES (25% with highest error)
num_to_cut = round(0.25 * length(detection_failed));
detection_failed(d_idx(1:num_to_cut)) = 1;

% Plot point correspondance for borderline cases
if 1
    figure(2);
    for t = d_idx(num_to_cut:num_to_cut+40)'
        img = imread(get_image_filename(index, t, db_dir));
        imshow(img); hold on;
        plot(lm_all(t,1:2:end),lm_all(t,2:2:end),'or');
        plot(pts(t,1:2:end),pts(t,2:2:end),'xb');
        for i = 1 : (size(lm,2)-1) / 2
            plot([pts(t,k(i * 2 - 1)), lm(t, 2*i-1)], [pts(t,k(i * 2)) lm(t, 2*i)], 'c');
        end
        title(sprintf('d = %f', d(t)));
        hold off; axis image; axis ij;
        % uncomment the pause command or add breakpoint to view the samples
        %pause
    end
end

% Check neutral frontals and keep them (or drop whole person?)
candidates_delete = detection_failed;
candidates_delete_frontal = candidates_delete & frontal_samples;
neutral_frontal_to_delete = find((candidates_delete_frontal' & index(:,3) == 1));
if ~isempty(neutral_frontal_to_delete)
    for t = neutral_frontal_to_delete'
        img = imread(get_image_filename(index, t, db_dir));
        imshow(img); hold on;
        plot(lm(t,1:2:end),lm(t,2:2:end),'or');
        plot(pts(t,1:2:end),pts(t,2:2:end),'xb');
        for i = 1 : (size(lm,2)-1) / 2
            plot([pts(t,k(i * 2 - 1)), lm(t, 2*i-1)], [pts(t,k(i * 2)) lm(t, 2*i)], 'c');
        end
        title(sprintf('d = %f', d(t)));
        hold off; axis image; axis ij;
    end
    warning('Would need to delete neutral frontal face --> keep them!');
    candidates_delete_frontal(neutral_frontal_to_delete) = 0;
	candidates_delete(neutral_frontal_to_delete) = 0;
end



% Remove all corresponding images to a frontal image that will be
% deleted.
if (any(candidates_delete_frontal))
    % Identify subject and expression
    dist = index(candidates_delete_frontal,1);
    subj = index(candidates_delete_frontal,2);
    expr = index(candidates_delete_frontal,3);
    % Find all samples with subject, expression combination and
    % add them to the deletion list.
    for se = 1 : size(dist, 1);
        candidates_se = index(:,1) == dist(se) & index(:,2) == subj(se) & index(:,3) == expr(se);
        candidates_delete = candidates_delete | candidates_se';
    end
end

% Update landmarks (set -1 for samples selected for removal)
lm_new = lm_all;
lm_inval = -ones(sum(candidates_delete), size(lm_all, 2));
lm_new(candidates_delete,:) = lm_inval;

detection_failed_new = all(lm_new' == -1);
lm_new = horzcat((1:size(lm_new,1))', lm_new);

% Show histograms of selected data subset
figure(3);

%data = index(~detection_failed_new, 1);
%subplot(421); hist(data, unique(data));
%title('distance');

data = index(~detection_failed_new, 2);
subplot(423); hist(data, unique(data));
title('subject');

data = index(~detection_failed_new, 3);
subplot(424); hist(data, unique(data));
title('expression');

data = pose(:, 2);
subplot(425); hist(data, unique(data));
title('yaw orig');

data = pose(~detection_failed_new, 2);
subplot(426); hist(data, unique(data));
title('yaw new');

data = pose(:, 1);
subplot(427); hist(data, unique(data));
title('pitch orig');

data = pose(~detection_failed_new, 1);
subplot(428); hist(data, unique(data));
title('pitch new');

% show frontal face with landmark errors
frontal_with_error = find(candidates_delete_frontal);
if 1
    figure(5);
    for t = frontal_with_error
        img = imread(get_image_filename(index, t, db_dir));
        imshow(img); hold on;
        plot(lm_all(t,1:2:end),lm_all(t,2:2:end),'or'); 
        plot(pts(t,1:2:end),pts(t,2:2:end),'xb');
        for i = 1 : (size(lm,2)-1) / 2
            plot([pts(t,k(i * 2 - 1)), lm(t, 2*i-1)], [pts(t,k(i * 2)) lm(t, 2*i)], 'c');
        end
        hold off; axis image; axis ij;
        title(sprintf('%s: d = %f', get_image_filename(index, t, db_dir), d(t)),'Interpreter','none');
        % uncomment the pause command or add breakpoint to view the samples
        %pause
    end
end

% number of samples after removal
num_samples = sum(~detection_failed_new)

% write landmarks table
dlmwrite(fullfile(db_dir, 'landmarks_dlib2.table'), lm_new, 'delimiter', '\t', 'precision', 7);
