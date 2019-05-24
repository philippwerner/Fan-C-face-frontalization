% This file is part of https://github.com/philippwerner/Fan-C-face-frontalization
% Copyright (c) 2013, Georgios Tzimiropoulos
%               2019, Philipp Werner
% License: BSD 2-Clause License (see LICENSE file in root directory)
% This code is based on: https://www.mathworks.com/matlabcentral/fileexchange/44651-active-appearance-models-aams

function [ img_out ] = warp_img( s_img, s_coord, t_coord, vis, symmetry, t_texture, t_triangles, size_out )

    if size(s_coord,1) ~= 1 || mod(size(s_coord,2), 2) ~= 0
        error('invalid input');
    end
    
    use_blending = ~all(vis == 0 | vis == 1);
    
    if isempty(vis)
        vis = true(size(s_coord));
    end
    if isempty(symmetry)
        symmetry = 1:length(s_coord);
    end

    num_of_triangles = size(t_triangles, 1);

    s_coord_x = s_coord(:,1:2:end);
    s_coord_y = s_coord(:,2:2:end);
    t_coord_u = t_coord(:,1:2:end);
    t_coord_v = t_coord(:,2:2:end);

    A = zeros(1,6); % affine transformation for each triangle

    % assigns each coordinate in warped image the corresponding (subpixel)
    % coordinates in original image
    warped_x = NaN(size_out(1), size_out(2));
    warped_y = NaN(size_out(1), size_out(2));
    
    if use_blending
        warped_x_blend = NaN(size_out(1), size_out(2));
        warped_y_blend = NaN(size_out(1), size_out(2));
        opacity = NaN(size_out(1), size_out(2));
    end
    
    
    % to speed-up the search, calculate bounding boxes of triangles
    minU = max(floor(min(t_coord_u(t_triangles),[],2)), 1);
    maxU = min(ceil(max(t_coord_u(t_triangles),[],2)), size(t_texture, 2));
    minV = max(floor(min(t_coord_v(t_triangles),[],2)), 1);
    maxV = min(ceil(max(t_coord_v(t_triangles),[],2)), size(t_texture, 1));
    
    for t = 1 : num_of_triangles
        t_cur_tri = t_triangles(t,:);
        
        %%% Step 1:
        % Coordinates of the three vertices of each triangle in base shape
        U = t_coord_u(t_cur_tri);
        V = t_coord_v(t_cur_tri);
        vis_cur_tri = vis(t_cur_tri);

        % Check if all coordinates of current triangle are visible
        if any(vis_cur_tri < 1)
            % At least one coordinate is not fully visible -> use mirror triangle
            X = s_coord_x(symmetry(t_cur_tri));
            Y = s_coord_y(symmetry(t_cur_tri));
        else % all visible
            % Coordinates of the three vertices of the corresponding triangle in current shape
            X = s_coord_x(t_cur_tri);
            Y = s_coord_y(t_cur_tri);
        end
        
        % Compute A from the U,V and X,Y
        denominator = (U(2) - U(1)) * (V(3) - V(1)) - (V(2) - V(1)) * (U(3) - U(1));

        A(1) = X(1) + ((V(1) * (U(3) - U(1)) - U(1)*(V(3) - V(1))) * (X(2) - X(1)) + (U(1) * (V(2) - V(1)) - V(1)*(U(2) - U(1))) * (X(3) - X(1))) / denominator;
        A(2) = ((V(3) - V(1)) * (X(2) - X(1)) - (V(2) - V(1)) * (X(3) - X(1))) / denominator;
        A(3) = ((U(2) - U(1)) * (X(3) - X(1)) - (U(3) - U(1)) * (X(2) - X(1))) / denominator;

        A(4) = Y(1) + ((V(1) * (U(3) - U(1)) - U(1) * (V(3) - V(1))) * (Y(2) - Y(1)) + (U(1) * (V(2) - V(1)) - V(1)*(U(2) - U(1))) * (Y(3) - Y(1))) / denominator;
        A(5) = ((V(3) - V(1)) * (Y(2) - Y(1)) - (V(2) - V(1)) * (Y(3) - Y(1))) / denominator;
        A(6) = ((U(2) - U(1)) * (Y(3) - Y(1)) - (U(3) - U(1)) * (Y(2) - Y(1))) / denominator;


        %%% Step 2
        % Get coordinates of all pixels within each triangle
        [v, u] = find(t_texture(minV(t):maxV(t),minU(t):maxU(t)) == t);
        v = v + (minV(t) - 1);
        u = u + (minU(t) - 1);

        if (~isempty(v) && ~isempty(u))

            ind_base = v + (u-1) * size_out(1);

            %%% Step 3
            warped_x(ind_base) = A(1) + A(2) .* u + A(3) .* v;
            warped_y(ind_base) = A(4) + A(5) .* u + A(6) .* v;
        end
        
        % at least one vertex is not fully visible
        if use_blending && any(vis_cur_tri < 1)
            % Coordinates of the three vertices of the corresponding triangle in current shape
            X = s_coord_x(t_cur_tri);
            Y = s_coord_y(t_cur_tri);
            
            % Compute A from the U,V and X,Y
            denominator = (U(2) - U(1)) * (V(3) - V(1)) - (V(2) - V(1)) * (U(3) - U(1));

            A(1) = X(1) + ((V(1) * (U(3) - U(1)) - U(1)*(V(3) - V(1))) * (X(2) - X(1)) + (U(1) * (V(2) - V(1)) - V(1)*(U(2) - U(1))) * (X(3) - X(1))) / denominator;
            A(2) = ((V(3) - V(1)) * (X(2) - X(1)) - (V(2) - V(1)) * (X(3) - X(1))) / denominator;
            A(3) = ((U(2) - U(1)) * (X(3) - X(1)) - (U(3) - U(1)) * (X(2) - X(1))) / denominator;

            A(4) = Y(1) + ((V(1) * (U(3) - U(1)) - U(1) * (V(3) - V(1))) * (Y(2) - Y(1)) + (U(1) * (V(2) - V(1)) - V(1)*(U(2) - U(1))) * (Y(3) - Y(1))) / denominator;
            A(5) = ((V(3) - V(1)) * (Y(2) - Y(1)) - (V(2) - V(1)) * (Y(3) - Y(1))) / denominator;
            A(6) = ((U(2) - U(1)) * (Y(3) - Y(1)) - (U(3) - U(1)) * (Y(2) - Y(1))) / denominator;
            
            % Get coordinates of all pixels within each triangle
            [v, u] = find(t_texture(minV(t):maxV(t),minU(t):maxU(t)) == t);
            v = v + (minV(t) - 1);
            u = u + (minU(t) - 1);
            
            if (~isempty(v) && ~isempty(u))

                ind_base = v + (u-1) * size_out(1);

                %%% Step 3
                warped_x_blend(ind_base) = A(1) + A(2) .* u + A(3) .* v;
                warped_y_blend(ind_base) = A(4) + A(5) .* u + A(6) .* v;
                % calculate opacity
                F = scatteredInterpolant(double(U)', double(V)', 1-double(vis_cur_tri)');
                opacity(ind_base) = F(u, v);
            end
            
        end

    end

    valid = ~isnan(warped_x(:));
    s_img = double(s_img);
    if ndims(s_img) == 2
        % grayscale
        img_out = zeros(size_out(1), size_out(2)) + 255;
        %img_out(valid) = interp2(s_img, warped_x(valid), warped_y(valid), 'nearest');
        img_out(valid) = interp2(s_img, warped_x(valid), warped_y(valid), 'linear');
        if use_blending
            if 1
                figure(124); colormap gray;
                subplot(221);
                imagesc(img_out);
                subplot(222);
                imagesc(opacity);
                hold on;
                triplot(t_triangles, t_coord(:,1:2:end), t_coord(:,2:2:end));
                hold off;
            end
            to_blend = ~isnan(warped_x_blend(:));
            o = opacity(to_blend);
            b = interp2(s_img, warped_x_blend(to_blend), warped_y_blend(to_blend), 'linear');
            img_out(to_blend) = (o) .* img_out(to_blend) + (1-o) .* b;
            if 1
                subplot(223);
                img2 = zeros(size(opacity));
                img2(to_blend) = b;
                imagesc(img2);
                subplot(224);
                imagesc(img_out);
                figure(1);
            end
        end
    else
        % rgb
        c1 = zeros(size_out(1), size_out(2)) + 255;
        c2 = c1;
        c3 = c1;
        c1(valid) = interp2(s_img(:,:,1), warped_x(valid), warped_y(valid), 'linear');
        c2(valid) = interp2(s_img(:,:,2), warped_x(valid), warped_y(valid), 'linear');
        c3(valid) = interp2(s_img(:,:,3), warped_x(valid), warped_y(valid), 'linear');
        if use_blending
            to_blend = ~isnan(warped_x_blend(:));
            o = opacity(to_blend);
            b1 = interp2(s_img(:,:,1), warped_x_blend(to_blend), warped_y_blend(to_blend), 'linear');
            b2 = interp2(s_img(:,:,2), warped_x_blend(to_blend), warped_y_blend(to_blend), 'linear');
            b3 = interp2(s_img(:,:,3), warped_x_blend(to_blend), warped_y_blend(to_blend), 'linear');
            c1(to_blend) = (o) .* c1(to_blend) + (1-o) .* b1;
            c2(to_blend) = (o) .* c2(to_blend) + (1-o) .* b2;
            c3(to_blend) = (o) .* c3(to_blend) + (1-o) .* b3;
        end        
        img_out = cat(3, c1, c2, c3);
        if 0
            figure(124);
            subplot(211);
            imshow(uint8(img_out));
            subplot(212);
            if exist('opacity','var')
                imagesc(opacity);
                hold on;
            end
            triplot(t_triangles, t_coord(:,1:2:end), t_coord(:,2:2:end));
            hold off;
        end
    end
    img_out = uint8(img_out);
    
end

