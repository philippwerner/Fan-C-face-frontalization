% This file is part of https://github.com/philippwerner/Fan-C-face-frontalization
% Authors: Philipp Werner and Frerk Saxen
% License: BSD 2-Clause License (see LICENSE file in root directory)

% usage: define db_dir, db_prefix, db_landmarks
if ~exist('db_dir', 'var') || ~exist('db_fn_prefix', 'var') || ~exist('db_landmarks', 'var')
    error(sprintf('Please define following variables before running util_load_db:\n\tdb_dir\n\tdb_fn_prefix\n\tdb_landmarks = string intraface, dlib or dlib_inner'));
end

db_load_new_signature = strcat(db_dir, '|', db_fn_prefix, '|', db_landmarks);
if ~exist('db_load_signature', 'var') || ~strcmp(db_load_signature, db_load_new_signature)
    %% Not loaded yet ... load data!
    db_load_fn_prefix = strcat('data/', db_fn_prefix, '_', db_landmarks, '_');
    db_load_fn_raw = strcat(db_load_fn_prefix, 'db_raw.mat');
    if ~exist(db_load_fn_raw, 'file')
        %error('File %s not found! Run util_create_db_mat.m to create it!', db_load_fn_raw);
        util_create_db_mat;
    else
        load(db_load_fn_raw);
    end
    db_load_fn_prefix = strcat('data/', db_fn_prefix, '_', db_landmarks, '_');
    db_load_fn_norm = strcat(db_load_fn_prefix, 'db_norm.mat');
    if ~exist(db_load_fn_norm, 'file')
        %error('File %s not found! Run util_create_db_mat.m to create it!', db_load_fn_norm);
        util_create_db_mat;
    else
        load(db_load_fn_norm);
    end
        
    %% Some checks
    if ~strcmp(db_dir, db_raw.database_root_path) || ~strcmp(db_dir, db_norm.database_root_path)
        error('Variable db_dir inconsistent with database file.');
    end
    switch db_landmarks
        case 'intraface'
            db_load_lm_dim = 2*49;
        case {'dlib', 'dlib2'}
            db_load_lm_dim = 2*68;
        case {'dlib_inner', 'dlib2_inner'}
            db_load_lm_dim = 2*51;
        otherwise
            error('The variable db_landmarks must be a string containing: intraface, dlib or dlib_inner!');
    end
    if db_load_lm_dim ~= size(db_raw.lm, 2) ||  db_load_lm_dim ~= size(db_norm.lm, 2)
        error('Variable db_landmarks inconsistent with database file.');
    end
    
    db_load_signature = db_load_new_signature;
end

clear db_load_new_signature db_load_fn_prefix db_load_fn_raw db_load_fn_norm db_load_lm_dim;