% implementation of Freedman et al. "SRA: Fast Removal of General Multipath for
% ToF Sensors"
clc; clear; close all
folder = '~/ROS/DeepEnd2End/DeepToF_release_0.1/src/GAN/datasets/mpi_correction_corrnormamp2depth_albedoaug_zpass_test/test_real_all_calib_my_room/';
fn = dir([folder '*.mat']);
folder_out = 'SRA_results_all_real_all_calib/';
mkdir(folder_out);
isnoise = 0;
isvis = 0;

for ifn = 1:numel(fn) %[170,69,86,110,129]
    fprintf('%s...',fn(ifn).name);
    load([folder fn(ifn).name]);
    corr = permute(im_pair(:,:,1:end/2),[2,3,1]);
    corr = (corr*2-1)*pi;
    [h,w,~] = size(corr);
    dist_gt = permute(im_pair(1,:,end/2+1:end),[2,3,1]) * 10;
    dmin = 0.5*min(dist_gt(dist_gt>0)); % lower bound
    dmax = 1.5*max(dist_gt(dist_gt<Inf)); % upper bound

    d = dmin:0.01:dmax; 
    f = [45180000,37650000];
    %f = [40,70]*1e6;
    m = numel(d);
    c = 3e8;
    Phi = [cos(4*pi*d*f(1)/c); cos(4*pi*d*f(2)/c); sin(4*pi*d*f(1)/c); sin(4*pi*d*f(2)/c)];
    if ~isnoise
        C = eye(4);
    else
        tmp = randn(1000,4)*0.5+0;
        C = cov(tmp);
    end
    Q = [-1 -1 -1 -1; -1 -1 -1 1; -1 -1 1 -1; -1 -1 1 1;
        -1 1 -1 -1; -1 1 -1 1; -1 1 1 -1; -1 1 1 1;
        1 -1 -1 -1; 1 -1 -1 1; 1 -1 1 -1; 1 -1 1 1;
        1 1 -1 -1; 1 1 -1 1; 1 1 1 -1; 1 1 1 1];
    options = optimoptions('linprog','Algorithm','interior-point','Display','off');
    distSRA = zeros(h,w);
    xall = zeros(h,w,m);
    corr(:,:,[5])=[];
    parfor ih = 1:h
        for iw = 1:w
            v = squeeze(corr(ih,iw,:));
            if numel(find(isnan(v)==1))>0
                continue;
            end
            f = ones(m,1);
            if ~isnoise
                A = Q*C*Phi;
                b = Q*C*v + 0.05*sum(abs(v(:)));
            else
                A = Q*abs(C.^(-1/2))*Phi;
                b = Q*abs(C.^(-1/2))*v + 0.05*sum(abs(v(:)));
            end
            x = linprog(f,A,b,[],[],zeros(m,1),[],options);
            if numel(x) == m
                xall(ih,iw,:) = x;
            else
                continue;
            end
            if isvis
                figure(1);
                plot(x);
            end
            p = find(x>0);
            if numel(p) == 0
                distSRA(ih,iw) = 0; %invalid
            else
                po = p(1);
                for i = 1:numel(p)
                    if x(p(i))>0.01*max(x)
                        po = p(i);
                        break;
                    end
                end
                distSRA(ih,iw) = d(po);
            end
        end
    end
    pointCloudSRA = my_depthToPointCloud(distSRA,'no_calib');
    %pointCloudSRA = depthToPointCloud(distSRA,'tintin');
    depthSRA = pointCloudSRA(:,:,3)';
    pointCloudGT = my_depthToPointCloud(dist_gt,'no_calib');
    %pointCloudGT = depthToPointCloud(dist_gt,'tintin');
    depthGT = pointCloudGT(:,:,3)';
    save([folder_out fn(ifn).name],'depthSRA','distSRA');
    if isvis
        figure(2); imagesc([depthGT depthSRA]); axis image; colorbar;
        figure(3); plot(depthSRA(160,:)); hold on; plot(depthGT(160,:));
    end
    fprintf('done\n');
end
