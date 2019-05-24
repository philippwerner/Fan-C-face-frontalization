% This file is part of https://github.com/philippwerner/Fan-C-face-frontalization
% Authors: Philipp Werner and Frerk Saxen
% License: BSD 2-Clause License (see LICENSE file in root directory)

function [ tform ] = tform_fit_prealign_mouth( detector_coord_row, lm_mouth_template )
%tform_fit_prealign Fit prealignment based on intraface eyes

    coord_mat = reshape(detector_coord_row, 2, [])';
   
    switch size(coord_mat, 1)
        case 49
            % intraface
            lm_mouth = vertcat(mean(coord_mat([32],:), 1), mean(coord_mat([38],:),1));
        case 51
            % dlib inner points
            lm_mouth = vertcat(mean(coord_mat([32],:), 1), mean(coord_mat([38],:), 1));
        case 68
            % dlib / ibug numbering
            lm_mouth = vertcat(mean(coord_mat([32+17],:), 1), mean(coord_mat([38+17],:), 1));
    end
    
    if nargin < 2
        lm_mouth_template = [ -0.5 0 ; 0.5 0 ];
    end
    
    tform = fitgeotrans(lm_mouth, lm_mouth_template, 'NonreflectiveSimilarity');
    
end

