% This file is part of https://github.com/philippwerner/Fan-C-face-frontalization
% Authors: Philipp Werner and Frerk Saxen
% License: BSD 2-Clause License (see LICENSE file in root directory)

function [ X, Y ] = regfunc_fanc_create_samples( mode, detector_lm_s, corresp_pt_s, detector_lm_t, corresp_pt_t )
% reg_create_samples Assemble samples for training or testing the
% registration model
%
% [ X ] = reg_create_samples(detector_lm_s)
%   Create sample matrix X for testing (one row for each image, columns are
%   features).
%
% [ X, Y ] = reg_create_samples(detector_lm_s, corresp_pt_s, detector_lm_t, corresp_pt_t)
%   Create sample matrix X and response matrix Y for training. On X, see
%   above. Y has the same number of rows like X, but more columns.
%   TODO...
%

    %% select feature extraction mode
    if ischar(mode)
        mode = lower(mode);
        add_eye_aligned_points = ~isempty(strfind(mode, 'x'));
        add_mouth_aligned_points = ~isempty(strfind(mode, 'm'));
        add_affine_aligned_points = ~isempty(strfind(mode, 'a'));
        apply_pca = false;
        add_squares = ~isempty(strfind(mode, '2'));
        add_lm_prod = ~isempty(strfind(mode, 'y'));
        add_squares2 = false;
    elseif isstruct(mode)
        point_reg_mode = lower(mode.point_reg_mode);
        add_eye_aligned_points = ~isempty(strfind(point_reg_mode, 'x'));
        add_mouth_aligned_points = ~isempty(strfind(point_reg_mode, 'm'));
        add_affine_aligned_points = ~isempty(strfind(point_reg_mode, 'a'));
        add_squares = ~isempty(strfind(point_reg_mode, '2'));
        add_lm_prod = ~isempty(strfind(point_reg_mode, 'y'));

        %dim_reduction_mode = lower(mode.dim_red_mode);
        apply_pca = isfield(mode, 'pca_transform') && isfield(mode, 'pca_mean');
        
        if isfield(mode,'feat_exp_mode')
            feat_expansion_mode = lower(mode.feat_exp_mode);        
            add_squares2 = ~isempty(strfind(feat_expansion_mode, '2'));
        else
            add_squares2 = false;
        end
    else
        error('mode: invalid type');
    end
    

    n_images = size(detector_lm_s, 1);
    %n_detect_pts = size(detector_lm_s, 2) / 2;
    
   
    %% prepare response matrix
    if nargin > 2
        n_responses = size(corresp_pt_s, 2) + size(corresp_pt_t, 2);
        Y = zeros(n_images, n_responses, 'single');
    
        %i = 1; %n = 2*n_detect_pts;
        %Y(:, i:i+n-1) = detector_lm_t;
        i = 1; n = 0;
        
        i = i+n; n = size(corresp_pt_s, 2);
        Y(:, i:i+n-1) = corresp_pt_s;
        
        i = i+n; n = size(corresp_pt_t, 2);
        Y(:, i:i+n-1) = corresp_pt_t;

        %i = i+n; n = n_corr_pts;
        %Y(:, i:i+n-1) = corresp_visiblity;
    end
    
    %% prepare feature matrix
    if add_eye_aligned_points
        X = single(detector_lm_s);
    else
        X = single([]);
    end
    
    %% Add mouth-aligned points as features
    if add_mouth_aligned_points
        M = zeros(size(detector_lm_s),'single');
        for i = 1 : size(detector_lm_s, 1)
            pts = detector_lm_s(i,:);
            tform = tform_fit_prealign_mouth(pts);
            pts = tform_forward(pts, tform);
            if 0
                plot(pts(1:2:end), pts(2:2:end), '.');
            end
            M(i,:) = pts;
        end
        X = horzcat(X, M);
    end
    
    %% Add affine-aligned points as features
    if add_affine_aligned_points
        if size(detector_lm_s, 1) < 1000
            error('TODO: load mean face shape');
        end
        lm_mean_row = mean(detector_lm_s);
        M = zeros(size(detector_lm_s),'single');
        for i = 1 : size(detector_lm_s, 1)
            pts = detector_lm_s(i,:);
            tform = tform_fit_prealign_allaff(pts, lm_mean_row);
            pts = tform_forward(pts, tform);
            if 0
                plot(pts(1:2:end), pts(2:2:end), '.');
            end
            M(i,:) = pts;
        end
        X = horzcat(X, M);
    end

    %% Add mouth height
    if 0
        idx = [ 45 48 ];
        X1 = X(:,1:2:end);
        X2 = X(:,2:2:end);
        if 0
            figure(111);
            mx = mean(X1);
            my = mean(X2);
            plot(mx, my, 'x');
            hold on; plot(mx(idx), my(idx), 'b-'); hold off;
        end
        X1d = diff(X1(:,idx)')';
        X2d = diff(X2(:,idx)')';
        mh = sqrt(X1d .^ 2 + X2d .^ 2);
        X = horzcat(X, mh);
    end
    
    % Nonlinear feature expansion: add squares
    if add_squares
        X = horzcat(X, X .^ 2);
    end
    
    %% Apply PCA
    if apply_pca
        X = bsxfun(@minus, X, mode.pca_mean);
        X = X * mode.pca_transform;
    end
     
    %% Nonlinear feature expansion
    orig_X = X;
    if add_squares2
        X = horzcat(X, orig_X .^ 2);
    end
    if add_lm_prod
        lm_prod = orig_X(:,1:2:end) .* orig_X(:,2:2:end);
        X = horzcat(X, lm_prod);
    end
    
    %% Add cosine features
    if 0
        Xn = bsxfun(@minus, X, mean(X));
        Xn = bsxfun(@rdivide, Xn, std(Xn));
        %freq = [30 60 90 120 150] / 180 * pi;
        freq = [30 90 150] / 180 * pi;
        %freq = [90] / 180 * pi;
        X = Xn;
        for alpha = freq
            for beta = 0%[-45 0 45]
                Xnew = cos(alpha * Xn + beta);
                X = horzcat(X, Xnew);
            end
        end
    end
        
end

