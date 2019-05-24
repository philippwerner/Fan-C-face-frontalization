% This file is part of https://github.com/philippwerner/Fan-C-face-frontalization
% Authors: Philipp Werner and Frerk Saxen
% License: BSD 2-Clause License (see LICENSE file in root directory)

classdef FaceRegistrationDatabase
    %FaceRegistrationDatabase Class of database for FaN-C face normalization
    
    properties
        sample_id,
        dist,
        subject_id,
        expression_id,
        seq_id,
        frame_id,
        pose,
        lm,             % landmarks
        pt,             % correspondence point coordinates
        pt_vis,         % correspondence point visibilities
        corresponding_frontal_idx,
        triangles,
        mesh_symm_corresp,
        database_root_path,
        xm2_feat
    end
    
    methods
        function obj = FaceRegistrationDatabase(index_, pose_, lm_, pt_, pt_vis_, remove_mask)
            % Check input
            n_samples = size(index_,1);
            if n_samples ~= size(pose_,1) || n_samples ~= size(lm_,1) || ...
                    n_samples ~= size(pt_,1) || n_samples ~= size(pt_vis_,1)
                error('Invalid input!');
            end
            if (size(pt_,2) - 1) / 2 ~= size(pt_vis_,2)-1
                error('Invalid input!');
            end
            if nargin < 6
                remove_mask = [];
            end
            
            % Identify samples with failed intraface detection
            detection_failed = all(lm_(:,2:end) == -1, 2);
            
            % Delete unwanted samples
            if isempty(remove_mask)
                remove_mask = detection_failed;
            else
                %remove_mask = remove_mask | detection_failed;
                remove_mask = detection_failed;
            end
            index_(remove_mask,:) = [];
            pose_(remove_mask,:) = [];
            lm_(remove_mask,:) = [];
            pt_(remove_mask,:) = [];
            pt_vis_(remove_mask,:) = [];
            invalid_count = sum(remove_mask);
            
            
            % Initialize properties
            obj.sample_id = uint32(index_(:,1));
            obj.dist = uint8(index_(:,2));
            obj.subject_id = uint16(index_(:,3));
            obj.expression_id = uint16(index_(:,4));
            obj.seq_id = uint16(index_(:,5));
            obj.frame_id = uint16(index_(:,6));
            obj.pose = single(pose_(:,2:end));
            obj.lm = single(lm_(:,2:end)) + 1;    % matlab images start at pixel (1,1)
            obj.pt = single(pt_(:,2:end)) + 1;    % matlab images start at pixel (1,1)
            obj.pt_vis = logical(pt_vis_(:,2:end));
            
            % assign each face a corresponding frontal face
            corr_idx = zeros(size(index_,1), 1, 'uint32');
            [~,~,ia] = unique(horzcat(obj.dist, obj.subject_id, obj.expression_id), 'rows');
            frontal_idx = sum(abs(obj.pose),2) == 0;
            for i = 1:max(ia)
                group_idx = ia == i;
                corr_sample_idx = find(frontal_idx & group_idx);
                if isempty(corr_sample_idx)
                    error('Corresponping frontal image is missing!');
                end
                corr_idx(group_idx) = corr_sample_idx;
            end
            obj.corresponding_frontal_idx = corr_idx;
            
            % check that all neutral frontal images are available
            neutral_frontal_idx = obj.expression_id == 1 & frontal_idx;
            if sum(neutral_frontal_idx) ~= length(unique(obj.subject_id)) * length(unique(obj.dist))
                error('Neutral expression frontal image missing for some subjects');
            end

            
            % Calculate triangles for the frontal unposed subject.
            mesh = double([obj.pt(1,1:2:end)', obj.pt(1,2:2:end)']);
            obj.triangles = delaunay(mesh(:,1),mesh(:,2));

            % Reorder triangles (from chin to forhead)
            triangles_center = zeros(size(obj.triangles, 1),2);
            for t = 1 : size(obj.triangles, 1)
                triangles_center(t,:) = mean(mesh(obj.triangles(t,:),:));
            end
            [~, I] = sort(triangles_center(:,2),'descend');
            obj.triangles = obj.triangles(I,:);            
            
            % Get the symmetry point for each mesh point to handle
            % occlusion
            x_mirrored = 500 - mesh(:,1);

            [~, I] = util_pdist2_fast(mesh, horzcat(x_mirrored, mesh(:,2)), 'euclidean', 'Smallest', 1); 
            obj.mesh_symm_corresp = I;

            % Set default database root path
            obj.database_root_path = '../database';
            
            obj.xm2_feat = [];
        end
        
        function fn = get_image_filename(obj, sample_id)
            i = sample_id;
            dist_str = sprintf('D%d', obj.dist(i));
            subj_str = sprintf('SUBJ%02d', obj.subject_id(i));
            expr_str = sprintf('EXP%02d', obj.expression_id(i));
            seq_str = sprintf('SEQ%02d', obj.seq_id(i));
            fn = sprintf('%s/%s/%s/%s/%s/%s_%s_%s_%s_%04d.jpg',...
                obj.database_root_path, ...
                dist_str, subj_str, expr_str, seq_str, ...
                dist_str, subj_str, expr_str, seq_str, obj.frame_id(i));
        end
        
        function corresp_target_idx = get_corresponding_frontal_idx(obj, task, train_sample_idx)
            switch (task)
                case 'PoseNorm'
                    corresp_target_idx = obj.corresponding_frontal_idx;
                case 'PoseIdentityNorm'
                    % reduce inter-person variablility (map every person to mean/first person in training set)
                    if nargin < 3
                        fsid = 1;
                    else
                        train_subj_ids = obj.subject_id(train_sample_idx);
                        fsid = min(train_subj_ids);
                    end
                    corresp_target_idx = zeros(size(obj.corresponding_frontal_idx));
                    first_person_frontal_idx = obj.subject_id == fsid & sum(abs(obj.pose),2) == 0; % & obj.dist == 80;
                    for expr_id = unique(obj.expression_id)'
                        cur_expr_idx = obj.expression_id == expr_id;
                        cur_expr_fpf_idx = find(cur_expr_idx & first_person_frontal_idx);
                        if length(cur_expr_fpf_idx) == 2
                            cur_expr_fpf_idx = find(cur_expr_idx & first_person_frontal_idx & obj.dist == 80);
                        end
                        corresp_target_idx(cur_expr_idx) = cur_expr_fpf_idx;
                    end
                case 'PoseExpressionNorm'
                    % reduce expression variablility (map every expression to neutral person)
                    corresp_target_idx = zeros(size(obj.corresponding_frontal_idx));
                    neutral_frontal_idx = obj.expression_id == 1 & sum(abs(obj.pose),2) == 0; % & obj.dist == 80;
                    for subj_id = unique(obj.subject_id)'
                        cur_subj_idx = obj.subject_id == subj_id;
                        cur_subj_nf_idx = find(cur_subj_idx & neutral_frontal_idx);
                        corresp_target_idx(cur_subj_idx) = cur_subj_nf_idx;
                    end
            end
        end
        
        function [ gt_pt_vis, gt_pt_vis2 ] = get_visibility_masks(obj)
            gt_pt_vis = logical(obj.pt_vis);
            % repeat each column 2 times
            gt_pt_vis2 = reshape(repmat(gt_pt_vis,2,1),size(gt_pt_vis,1),2*size(gt_pt_vis,2));
        end
        
        function plot_image(obj, sample_idx, figure_idx)
            fn = get_image_filename(obj, sample_idx);
            figure(figure_idx);
            imagesc(imread(fn)); colormap('gray'); axis('ij'); axis('image');
        end
        
        function plot_visibility_points(obj, sample_idx, figure_idx)
            vis = obj.pt_vis(sample_idx, :);
            vis = [vis;vis];
            vis = vis(:)';
            vis_points = obj.pt(sample_idx, vis);
            invis_points = obj.pt(sample_idx, ~vis);
            figure(figure_idx);
            plot(vis_points(1:2:end),vis_points(2:2:end),'.b', invis_points(1:2:end),invis_points(2:2:end),'.r');
        end
     end
    
end

