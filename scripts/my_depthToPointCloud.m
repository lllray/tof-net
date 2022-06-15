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
    elseif  strcmp(cameratype,'0428')
        fx = 220;
        fy = 220;
        cx = 120;
        cy = 86;
        k1 = 0;
        k2 = 0;
        k3 = 0;
        p1 = 0;
        p2 = 0;
%         k1 = 0.0427;
%         k2 = -0.0413;
%         k3 = 0;
%         p1 = 0.00114;
%         p2 = 0.00038;
    elseif  strcmp(cameratype,'0429')
        fx = 224;
        fy = 224;
        cx = 118;
        cy = 84;
        k1 = 0;
        k2 = 0;
        k3 = 0;
        p1 = 0;
        p2 = 0;
%         k1 = 0.0427;
%         k2 = -0.0413;
%         k3 = 0;
%         p1 = 0.00114;
%         p2 = 0.00038;
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

    % depthToPointCloud
    pointCloudFrame = zeros(width,height,3);
%     load('/Users/ssu/GitHub/LearningMultiPath/local/data_tintin/meas_0926/0926_depth_39.mat');
    distances = depth;
    for u = 1:width
        for v = 1:height
            pointCloudFrame(u,v,1) = (u - cx)/fx * distances(v,u);
            pointCloudFrame(u,v,2) = (v - cy)/fy * distances(v,u);
            pointCloudFrame(u,v,3) = distances(v,u);
        end
    end
end