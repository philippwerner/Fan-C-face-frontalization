% This file is part of https://github.com/philippwerner/Fan-C-face-frontalization
% Authors: Philipp Werner and Frerk Saxen
% License: BSD 2-Clause License (see LICENSE file in root directory)

function [ tform, scale ] = tform_fit_prealign( detector_coord_row, lm_eyes_template )
%tform_fit_prealign Fit prealignment based on intraface eyes

    coord_mat = reshape(detector_coord_row, 2, [])';
   
    switch size(coord_mat, 1)
        case 49
            % intraface
            lm_eyes = vertcat(mean(coord_mat([20 23],:)), mean(coord_mat([26 29],:)));
        case 51
            % dlib inner points
            lm_eyes = vertcat(mean(coord_mat([20 23],:)), mean(coord_mat([26 29],:)));
        case 68
            % dlib / ibug numbering
            lm_eyes = vertcat(mean(coord_mat([37 40],:)), mean(coord_mat([43 46],:)));
    end
    
    if nargin < 2
        lm_eyes_template = [ -0.5 -0.5 ; 0.5 -0.5 ];
        %lm_eyes_template = [ 0.4 0.3 ; 0.6 0.3 ];
    end
    
    tform = fitgeotrans(lm_eyes, lm_eyes_template, 'NonreflectiveSimilarity');
    
    if nargout > 1
        d = norm(lm_eyes(1,:) - lm_eyes(2,:));
        dt = norm(lm_eyes_template(1,:) - lm_eyes_template(2,:));
        scale = dt / d;
    end

end

