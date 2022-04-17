clc;clear;%close all
% load('~/ROS/DeepEnd2End/DeepToF_release_0.1/src/GAN/results/my_resnet_lite_my_imageGAN_128_pretrained/latest_test/images/fake_B/000001.mat') %results are stored as .mat files under [results_folder]/fake_B
% load('~/ROS/DeepEnd2End/DeepToF_release_0.1/src/GAN/results/my_resnet_lite_my_imageGAN_128_pretrained/latest_test_real_all_calib/images/fake_B/1110_pix2pix_test_real_5.mat') %results are stored as .mat files under [results_folder]/fake_B
mydata = '0417-2';
load(sprintf('~/ROS/DeepEnd2End/DeepToF_release_0.1/src/GAN/results/my_resnet_lite_my_imageGAN_128_pretrained/latest_test_real_all_calib/images/fake_B/%s_pix2pix_test_real_0.mat',mydata)) %results are stored as .mat files under [results_folder]/fake_B
gt = single(imread(sprintf('~/bag/tintin_EE367/my_data/%s/phase_calibrated_norm2amp_rebuttal_mean5/%s_depth_0.png',mydata,mydata)))/1e4;
x = x*10 %scale from [0,1] back to distance in meters
result = squeeze(x(1,:,:));
mkdir(['~/ROS/DeepEnd2End/DeepToF_release_0.1/src/GAN/results/' mydata]);
imwrite(result, sprintf('~/ROS/DeepEnd2End/DeepToF_release_0.1/src/GAN/results/%s/this_depth.png',mydata),'Bitdepth',16);
imagesc(squeeze(x(1,:,:))) %display tofnet output
%error = gt.-result;

figure(1);
suptitle('result');
subplot(131); imagesc(gt); axis image; colorbar; title('gt');
subplot(132); imagesc(result); axis image; colorbar; title('result');
%subplot(133); imagesc(error); axis image; colorbar; title('error');

%%%%%%%%% visualize point clouds
%need sensor param
pointCloud_gt = my_depthToPointCloud(gt);
pointCloud_this = depthToPointCloud(result);

figure(2);
subplot(121); pcshow(reshape(pointCloud_gt,240*180,3)); title('point cloud of gt depth');
subplot(122); pcshow(reshape(pointCloud_this,320*240,3)); title('point cloud of this depth');