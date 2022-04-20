clc; clear; close all
data_type = 2;
if data_type == 1
 load('~/ROS/DeepEnd2End/DeepToF_release_0.1/src/GAN/datasets/mpi_correction_corrnormamp2depth_albedoaug_zpass/train/000980.mat');
%load('~/ROS/DeepEnd2End/DeepToF_release_0.1/src/GAN/datasets/mpi_correction_corrnormamp2depth_albedoaug_zpass/test_real_all_calib_new/train/1110_pix2pix_test_real_17.mat');
else load('~/bag/tintin_EE367/my_data/dataset-0420-6/train/dataset-0420-6_pix2pix_test_real_66.mat');
end
dist_gt = permute(im_pair(1,:,end/2+1:end),[2,3,1]) * 10;

Imat1 = permute(im_pair(1,:,1:end/2),[2,3,1]);
Qmat1 = permute(im_pair(2,:,1:end/2),[2,3,1]);
Imat2 = permute(im_pair(3,:,1:end/2),[2,3,1]);
Qmat2 = permute(im_pair(4,:,1:end/2),[2,3,1]);
Amp = permute(im_pair(5,:,1:end/2),[2,3,1]);


figure(1);
suptitle('impair');
subplot(321); imagesc(dist_gt); axis image; colorbar; title('gt');
subplot(322); imagesc(Amp); axis image; colorbar; title('Amp');
subplot(323); imagesc(Imat1); axis image; colorbar; title('Imat1');
subplot(324); imagesc(Qmat1); axis image; colorbar; title('Qmat1');
subplot(325); imagesc(Imat2); axis image; colorbar; title('Imat2');
subplot(326); imagesc(Qmat2); axis image; colorbar; title('Qmat2');
% 
%%%%%%%%% visualize point clouds
%need sensor param
if data_type == 1
    pointCloud_gt = depthToPointCloud(dist_gt);
    figure(2);
    pcshow(reshape(pointCloud_gt,320*240,3)); title('point cloud of gt depth');
elseif data_type == 2
    pointCloud_gt = my_depthToPointCloud(dist_gt,'no_calib');
    figure(2);
    pcshow(reshape(pointCloud_gt,240*180,3)); title('point cloud of gt depth');
end