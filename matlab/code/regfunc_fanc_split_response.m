% This file is part of https://github.com/philippwerner/Fan-C-face-frontalization
% Authors: Philipp Werner and Frerk Saxen
% License: BSD 2-Clause License (see LICENSE file in root directory)

function [  detector_lm_t, corresp_pt_s, corresp_pt_t ] = regfunc_fanc_split_response( Y, n_corr_dim )
%regfunc_fanc_split_response Split response of pt regression model(s)

    %n_detect_dim = 2*49;
    %n_corr_dim = (size(Y,2) - n_detect_dim) / 2;
    if nargin < 2
        n_corr_dim = size(Y,2) / 2;
    end
    
    %i = 1; n = n_detect_dim;
    %detector_lm_t = Y(:, i:i+n-1);
    i = 1; n = 0;
    detector_lm_t = [];

    i = i+n; n = n_corr_dim;
    corresp_pt_s = Y(:, i:i+n-1);

    i = i+n;
    corresp_pt_t = Y(:, i:end);

end

