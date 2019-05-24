% This file is part of https://github.com/philippwerner/Fan-C-face-frontalization
% Copyright (c) 2013, Georgios Tzimiropoulos
%               2019, Philipp Werner
% License: BSD 2-Clause License (see LICENSE file in root directory)
% This code is based on: https://www.mathworks.com/matlabcentral/fileexchange/44651-active-appearance-models-aams

function base_texture = warp_create_texture_base(vertices, triangles, resolution)
% warp_create_texture_base Create base_texture to warp image
base_texture = zeros(resolution(1), resolution(2));
vertices = vertcat(vertices(:,1:2:end), vertices(:,2:2:end));
for i = 1:size(triangles,1)
    % vertices for each triangle
    X = vertices(1, triangles(i,:));
    Y = vertices(2, triangles(i,:));
    % calculate ROI
    minX = floor(min(X)); maxX = ceil(max(X));
    minY = floor(min(Y)); maxY = ceil(max(Y));
    szX = maxX - minX + 1; szY = maxY - minY + 1;
    % mask for each traingle
    mask = poly2mask(X(:)-minX+1, Y(:)-minY+1, szY, szX) .* i;
    % assemble complete base texture
    if min(minX, minY) >= 1 && maxY <= size(base_texture, 1) && maxX <= size(base_texture, 2)
        base_texture(minY:maxY, minX:maxX) = max(base_texture(minY:maxY, minX:maxX), mask);
    else
        dx1 = max(-minX + 1, 0); dx2 = min(size(base_texture, 2) - maxX, 0);
        dy1 = max(-minY + 1, 0); dy2 = min(size(base_texture, 1) - maxY, 0);
        bt_idx_x = minX+dx1 : maxX+dx2;
        bt_idx_y = minY+dy1 : maxY+dy2;
        base_texture(bt_idx_y, bt_idx_x) = ...
            max( base_texture( bt_idx_y, bt_idx_x ), ...
                 mask( 1+dy1:end+dy2, 1+dx1:end+dx2 ) );
    end
end