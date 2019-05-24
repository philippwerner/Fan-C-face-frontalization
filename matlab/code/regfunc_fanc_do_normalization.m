% This file is part of https://github.com/philippwerner/Fan-C-face-frontalization
% Authors: Philipp Werner and Frerk Saxen
% License: BSD 2-Clause License (see LICENSE file in root directory)

function [ to_pred_img, interm_results ] = ...
    regfunc_fanc_do_normalization( sr_in_img, sr_in_lm, model)
%regfunc_fanc_do_normalization Normalize face image with Fan-C method.

    % pre-alignment (raw image coordinates to normalized coordinates and vica versa)
    sr_sn_tform = tform_fit_prealign(sr_in_lm);
    sn_in_lm = tform_forward(sr_in_lm, sr_sn_tform);

    % prepare features for prediction
    sn_in_feat_vec = regfunc_fanc_create_samples(model.feature_mode, sn_in_lm);
    dataset = libDataset.create_dataset(sn_in_feat_vec, zeros(1, length(model.pt_model.predictor_idx)));
    dataset = libDataset.normalize(dataset, model.feature_norm);
    
    % predict corresp. points' coordinates
    prediction = double(libML.predict(dataset, model.pt_model));
    [ ~, sn_pred_pt, tn_pred_pt ] = regfunc_fanc_split_response(prediction);
    
    % predict visibility and binarize (for network)
    sn_pred_vis = double(libML.predict(dataset, model.vis_model));
    sn_pred_vis = double(sn_pred_vis > 0.5);
    
    % blending
    use_blending = true;
    if use_blending
        tri_vert = model.warp_triangles;
        % look at triangles with one invisible vertex
        tri_vis = sn_pred_vis(tri_vert);
        mixed_tri = ~all(tri_vis == 0,2) & ~all(tri_vis == 1,2);
        mixed_tri_vert = tri_vert(mixed_tri,:);
        % set visible vertices to partially visible
        mixed_tri_visible_vert = mixed_tri_vert(sn_pred_vis(mixed_tri_vert) == 1);
        sn_pred_vis(mixed_tri_visible_vert) = 0.0001; % 0.0001
        % look at triangles with partially visible vertex and visible vertex
        tri_vis = sn_pred_vis(tri_vert);
        mixed_tri = any(tri_vis == 1,2) & any(tri_vis < 1,2);
        mixed_tri_vert = tri_vert(mixed_tri,:);
        % set visible vertices to partially visible
        mixed_tri_visible_vert = mixed_tri_vert(sn_pred_vis(mixed_tri_vert) == 1);
        sn_pred_vis(mixed_tri_visible_vert) = 0.5; % 0.33
    end
    
    % visibility post-processing: set that of the more visible half to visible
    if 1
        lv = sum(sn_pred_vis(model.left_pt_idx));
        rv = sum(sn_pred_vis(model.right_pt_idx));
        if lv > rv
            sn_pred_vis(model.left_pt_idx) = 1;
        else
            sn_pred_vis(model.right_pt_idx) = 1;
        end
    end

    % backproject corresp. points to original image
    sr_pred_pt = tform_backward(sn_pred_pt, sr_sn_tform);
    
    % project normalized to output image coordinates
    if isfield(model, 'to_size')
        if isfield(model, 'to_eye_dist')
            if isfield(model, 'to_eye_y')
                [tn_to_tform, to_size] = tform_norm_to_output(model.to_size, model.to_eye_dist, model.to_eye_y);
            else
                [tn_to_tform, to_size] = tform_norm_to_output(model.to_size, model.to_eye_dist);
            end
        else
            [tn_to_tform, to_size] = tform_norm_to_output(model.to_size);
        end
    else
        [tn_to_tform, to_size] = tform_norm_to_output();
    end
    to_pred_pt = tform_forward(tn_pred_pt, tn_to_tform);

    % try to use GPU warper... or fall back to matlab implementation
    try
        gpu_warper = model.gpu_warper;
        to_pred_img = gpu_warper.warp(sr_in_img, ...
                sr_pred_pt, to_pred_pt, ...
                sn_pred_vis, model.warp_triangles, ...
                model.warp_pt_sym_corresp, to_size);
    catch
        % prepare triangle map for warping
        to_texture = warp_create_texture_base(to_pred_pt, model.warp_triangles, to_size);

        % warp texture
        to_pred_img = warp_img(sr_in_img, sr_pred_pt, to_pred_pt, sn_pred_vis, model.warp_pt_sym_corresp, to_texture, model.warp_triangles, to_size);
    end
    
    % add background
    if isfield(model, 'show_background') && model.show_background
        % warp background based on visible points
        vis_idx = sn_pred_vis == 1;
        to_pred_pt_xy = horzcat(to_pred_pt(1:2:end)',to_pred_pt(2:2:end)');
        sr_pred_pt_xy = horzcat(sr_pred_pt(1:2:end)',sr_pred_pt(2:2:end)');
        %tform_bg = fitgeotrans(sr_pred_pt_xy(vis_idx,:), to_pred_pt_xy(vis_idx,:), 'NonreflectiveSimilarity');
        tform_bg = fitgeotrans(sr_pred_pt_xy(vis_idx,:), to_pred_pt_xy(vis_idx,:), 'Affine');
        bg_img = imwarp(sr_in_img, tform_bg, 'OutputView', imref2d(to_size, [1 to_size(2)], [1 to_size(1)]));
        % show warped background
        if 0
            figure(12349);
            imshow(bg_img);
            hold on;
            scatter(to_pred_pt_xy(:,1), to_pred_pt_xy(:,2), (vis_idx+2)*5, vis_idx, 'g.');
            hold off;
        end
        % add background to frontalized image
        bg_idx = to_texture(:) == 0;
        bg_r = bg_img(:,:,1);
        bg_g = bg_img(:,:,2);
        bg_b = bg_img(:,:,3);
        fg_r = to_pred_img(:,:,1);
        fg_g = to_pred_img(:,:,2);
        fg_b = to_pred_img(:,:,3);
        fg_r(bg_idx) = bg_r(bg_idx);
        fg_g(bg_idx) = bg_g(bg_idx);
        fg_b(bg_idx) = bg_b(bg_idx);
        to_pred_img = cat(3, fg_r, fg_g, fg_b);
        %sn_pred_pt_row = horzcat(sn_pred_pt(1:2:end)', sn_pred_pt(2:2:end)');
        % calculate convex hull of visible vertices (we know the correct
        % coordinates in source and target for those)
        
%         DT = delaunayTriangulation(tn_pred_pt_x(vis_idx),tn_pred_pt_y(vis_idx));
%         k = convexHull(DT);
%         
%         figure(938457);
%         triplot(DT);
%         hold on;
%         scatter(tn_pred_pt_x, tn_pred_pt_y, (vis_idx+2)*5, vis_idx, '.');
%         plot(DT.Points(k,1), DT.Points(k,2), 'og');
%         hold off;
    end
    
    % return intermediate results
    if nargout > 1
        interm_results = struct;
        interm_results.sr_in_lm = sr_in_lm;
        interm_results.sn_in_lm = sn_in_lm;
        interm_results.sr_sn_tform = sr_sn_tform;
        interm_results.sn_pred_pt = sn_pred_pt;
        interm_results.tn_pred_pt = tn_pred_pt;
        interm_results.sr_pred_pt = sr_pred_pt;
        interm_results.to_pred_pt = to_pred_pt;
        interm_results.sn_pred_vis = sn_pred_vis;
    end
    
    % visualize intermediate results
    if 0
        figure(124);
        subplot(211); imshow(sr_in_img);
        hold on; plot(sr_in_lm(:,1:2:end),sr_in_lm(:,2:2:end), 'gx'); hold off;
        hold on; plot(sr_pred_pt(:,1:2:end),sr_pred_pt(:,2:2:end), '.'); hold off;
        subplot(212); imshow(to_pred_img);
        hold on; plot(to_pred_pt(:,1:2:end),to_pred_pt(:,2:2:end), '.'); hold off;
    end

end

