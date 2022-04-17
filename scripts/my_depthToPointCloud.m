% % code converted from depthToPointCloud.cpp of voxelsdk
function pointCloudFrame = my_depthToPointCloud(depth, cameratype)
    % params
    global width height fx fy cx cy k1 k2 k3 p1 p2
    width = 240;
    height = 180;
    if ~exist('cameratype','var') || strcmp(cameratype,'no_calib')
        fx = 224;
        fy = 224;
        cx = 120;
        cy = 90;
        k1 = 0;
        k2 = 0;
        k3 = 0;
        p1 = 0;
        p2 = 0;
    else
        fx = 224;
        fy = 224;
        cx = 113;
        cy = 88;
        k1 = 0.0381695;
        k2 = 0.084686;
        k3 = -0.000104;
        p1 = 0.007071;
        p2 = -0.575066;
    end
    if 1
        % init
        directions = zeros(height*width,3);
        for v = 1:height
            for u = 1:width
        %         [u_, v_] = screenToNormalizedScreen(u, v);
                [u_, v_] = lensCorrection(u, v);
                [px, py, pz] = normalizedScreenToUnitWorld(u_, v_);
                directions((v-1)*width+u,:) = [px, py, pz];
            end
        end
    else
        % load precomputed
        load(strcat('directions_',cameratype));
    end

    % depthToPointCloud
    pointCloudFrame = zeros(width,height,3);
%     load('/Users/ssu/GitHub/LearningMultiPath/local/data_tintin/meas_0926/0926_depth_39.mat');
    distances = depth;
    for u = 1:width
        for v = 1:height
            pointCloudFrame(u,v,1) = directions((v-1)*width+u,1) * distances(v,u);
            pointCloudFrame(u,v,2) = directions((v-1)*width+u,2) * distances(v,u);
            pointCloudFrame(u,v,3) = directions((v-1)*width+u,3) * distances(v,u);
        end
    end

    % visualize
%     figure;
%     pcshow(reshape(pointCloudFrame,320*240,3)); 
%     title('point cloud');

    % supporting functions

    function [xs, ys] = screenToNormalizedScreen(u, v)
%         global fx fy cx cy k1 k2 k3 p1 p2
        iters = 100;
        ys = (v - cy) / fy; yss = ys;
        xs = (u - cx) / fx; xss = xs;

        for j = 0:iters
            r2 = xs * xs + ys * ys;
            icdist = 1.0 / (1 + ((k3 * r2 + k2) * r2 + k1) * r2);
            deltaX = 2 * p1 * xs * ys + p2 * (r2 + 2 * xs * xs);
            deltaY = p1 * (r2 + 2 * ys * ys) + 2 * p2 * xs * ys;
            xs = (xss - deltaX)*icdist;
            ys = (yss - deltaY)*icdist;
        end
    end

    function [x__, y__] = lensCorrection(u, v)
%         global fx fy cx cy k1 k2 k3 p1 p2
        x_ = (u - cx)/fx;
        y_ = (v - cy)/fy;
        r2 = x_ * x_ + y_ * y_;
        r4 = r2 * r2;
        r6 = r2 * r4;
        x__ = x_ * (1.0 + k1 * r2 + k2 * r4 + k3 * r6) + 2.0 * p1 * x_ * y_ + p2 * (r2 + 2.0 * x_ * x_);
        y__ = y_ * (1.0 + k1 * r2 + k2 * r4 + k3 * r6) + p1 * (r2 + 2.0 * y_ * y_) + 2.0 * p2 * x_ * y_;
    end

    function [px,py,pz] = normalizedScreenToUnitWorld(u, v)
        norm = 1.0 / sqrt(u * u + v * v + 1.0);
        px = u * norm;
        py = v * norm;
        pz = norm;
    end

end