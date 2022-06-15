% % code converted from dispToPointCloud.cpp of voxelsdk
function pointCloudFrame = my_dispToPointCloud(disp, cameratype)
    % params
    global width height fx fy cx cy k1 k2 k3 p1 p2 fb
    width = 1280;
    height = 800;
    if  strcmp(cameratype,'33030')
        width = 1280;
        height = 800;
        fx = 938.06;
        fy = 938.65;
        cx = 653.59;
        cy = 393.76;
        fb = 0.1*fx
        k1 = 0.0;
        k2 = 0.0;
        k3 = 0.0;
        p1 = 0.0;
        p2 = 0.0;
    else 
        width = 1280;
        height = 800;
        fx = 938.06;
        fy = 938.65;
        cx = 653.59;
        cy = 393.76;
        fb = 0.1*fx
        k1 = 0.0;
        k2 = 0.0;
        k3 = 0.0;
        p1 = 0.0;
        p2 = 0.0;
    end

    pointCloudFrame = zeros(width,height,3);
    distances = fb./disp;
    for u = 1:width
        for v = 1:height
            pointCloudFrame(u,v,1) = ( u - cx ) / fx * distances(v,u);
            pointCloudFrame(u,v,2) = ( v - cy ) / fy * distances(v,u);
            pointCloudFrame(u,v,3) = distances(v,u);
        end
    end
end