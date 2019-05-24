% This file is part of https://github.com/philippwerner/Fan-C-face-frontalization
% Authors: Philipp Werner and Frerk Saxen
% License: BSD 2-Clause License (see LICENSE file in root directory)

function [ coord_row ] = tform_backward( coord_row, tform )
%tform_forward Apply forward transform on row coordinate vector
    
    coord_mat = reshape(coord_row, 2, [])';
    coord_mat = tform.transformPointsInverse(coord_mat);
    coord_row = reshape(coord_mat', 1, []);

end

