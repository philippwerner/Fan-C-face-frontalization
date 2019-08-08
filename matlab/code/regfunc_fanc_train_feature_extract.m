% This file is part of https://github.com/philippwerner/Fan-C-face-frontalization
% Authors: Philipp Werner and Frerk Saxen
% License: BSD 2-Clause License (see LICENSE file in root directory)

function [ mode ] = regfunc_fanc_train_feature_extract( mode, detector_lm_s )
% prepare feature extraction during training (PCA transform)
% different alignments -> concat -> PCA -> add squared or other

    if ischar(mode)
        % old style -> do nothing
        return;
    end
    
    if ~isstruct(mode)
        error('mode: invalid type');
    end
    
    point_reg_mode = lower(mode.point_reg_mode);
    dim_reduction_mode = strtrim(lower(mode.dim_red_mode));
    %feat_expansion_mode = lower(mode.feat_exp_mode);

    % parse dim reduction params
    if isempty(dim_reduction_mode)
        return;
    end
    if strcmp(dim_reduction_mode(1:3), 'pca') == 0
        error('dim_reduction_mode: parse error');
    end
    variance_to_keep = sscanf(dim_reduction_mode(4:end), '%f');
    if length(variance_to_keep) ~= 1
        error('dim_reduction_mode: parse error');
    end
    if variance_to_keep > 1
        variance_to_keep = variance_to_keep / 100;
    end
    
    % get registered point location features
    X = regfunc_ours_create_samples(point_reg_mode, detector_lm_s);
    
    % cancel out mean
    mode.pca_mean = mean(X, 1);
    X = bsxfun(@minus, X, mode.pca_mean);
    
    [pc,score,latent] = pca(double(X),'Centered',false);
    %score2 = signals * pc;
    
    comul_variances = cumsum(latent)./sum(latent);
    %comul_variances = comul_variances(1:80)'
    pcs_to_use = find(comul_variances >= variance_to_keep, 1, 'first');

    % Try to reconstruct features from reduced form and calculate error
    if 0
        score(:,(pcs_to_use+1):end) = 0;
        features_reconstruction = score / pc;
        err = abs(X - features_reconstruction);
        mean_error = mean(err(:))
        max_error = max(err(:))
        figure(10); hist(err(:),100)
        figure(11); hist(err(:)/range(X(:)),100);
    end

    pc = pc(:,1:pcs_to_use);
    if 0
        score2 = X * pc;
        score = score(:,1:pcs_to_use);
        max(abs(score(:)-score2(:)))
    end
    mode.pca_transform = pc;
    
end

