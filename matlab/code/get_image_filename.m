% This file is part of https://github.com/philippwerner/Fan-C-face-frontalization
% Authors: Philipp Werner and Frerk Saxen
% License: BSD 2-Clause License (see LICENSE file in root directory)

function fn = get_image_filename(index, sample_id, db_dir)
    i = sample_id;
    dist_str = sprintf('D%d', index(i,1));
    subj_str = sprintf('SUBJ%02d', index(i,2));
    expr_str = sprintf('EXP%02d', index(i,3));
    seq_str = sprintf('SEQ%02d', index(i,4));
    %fn = sprintf('%s/%s/%s/%s/%s/%s_%s_%s_%s_%04d_pt_check.jpg',...
    fn = sprintf('%s/%s/%s/%s/%s/%s_%s_%s_%s_%04d.jpg',...
        db_dir, ...
        dist_str, subj_str, expr_str, seq_str, ...
        dist_str, subj_str, expr_str, seq_str, index(i,5));
end
