% This file is part of https://github.com/philippwerner/Fan-C-face-frontalization
% Authors: Philipp Werner and Frerk Saxen
% License: BSD 2-Clause License (see LICENSE file in root directory)

function [ tform, img_size ] = tform_norm_to_output( img_size, img_eye_dist, img_y_eye )
%tform_norm_to_output Get transform from norm to output space
    
    if nargin < 1
        img_size = [200 180];   % image size in pixel
    end
    if nargin < 2
        img_eye_dist = 80/180 * img_size(2);    % new eye distance in pixel (scaling factor)
    end
    if nargin < 3
        img_y_eye = 65/200 * img_size(1);      % in pixel
    end
    
    img_size2 = img_size ./ 2;
    
    tform_mat = eye(3);
    tform_mat(1,1) = img_eye_dist;
    tform_mat(2,2) = img_eye_dist;
    tform_mat(3,1) = img_size2(2);
    tform_mat(3,2) = 0.5*img_eye_dist + img_y_eye;
    tform = affine2d(tform_mat);

end

