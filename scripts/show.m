clc;clear;close all;
% load('~/ROS/DeepEnd2End/DeepToF_release_0.1/src/GAN/results/my_resnet_lite_my_imageGAN_128_pretrained/latest_test/images/fake_B/000001.mat') %results are stored as .mat files under [results_folder]/fake_B
% load('~/ROS/DeepEnd2End/DeepToF_release_0.1/src/GAN/results/my_resnet_lite_my_imageGAN_128_pretrained/latest_test_real_all_calib/images/fake_B/1110_pix2pix_test_real_5.mat') %results are stored as .mat files under [results_folder]/fake_B
data_type = 2; % 1 = tintin 2 = my
model_name = 'datasets-indoor-mix_my_resnet_lite_my_imageGAN_128'; % my_resnet_lite_my_imageGAN_128_pretrained,22-04-20-num4,my_resnet_lite_my_imageGAN_128
data_name = "latest_test_real_all_calib_indoor_mix";
type = 'fake_B'; % fake_B,real_A,real_B,sra
if data_type == 1
mydata = '1110';
take = 5;
% mydata = '0926';
% take = 40;
elseif data_type == 2
mydata = 'dataset-0424-2';
take = 25;
end

if data_type == 1
    load(sprintf('~/ROS/DeepEnd2End/DeepToF_release_0.1/src/GAN/results/%s/%s/images/%s/%s_pix2pix_test_real_%d.mat',model_name,data_name,type,mydata,take)) %results are stored as .mat files under [results_folder]/fake_B
    gt = single(imread(sprintf('~/bag/tintin_EE367/data_tintin/meas_%s/phase_calibrated_norm2amp_rebuttal_mean5/%s_depth_%d.png',mydata,mydata,take)))/1e4;
elseif data_type == 2
    load(sprintf('~/ROS/DeepEnd2End/DeepToF_release_0.1/src/GAN/results/%s/%s/images/%s/%s_pix2pix_test_real_%d.mat',model_name,data_name,type,mydata,take)) %results are stored as .mat files under [results_folder]/fake_B
    load(sprintf('~/bag/tintin_EE367/my_data/%s/phase_calibrated_norm2amp_rebuttal_mean1/%s_depth_%d.mat',mydata,mydata,take));
    gt = depth;
end
if strcmp(type,'sra') == 1
    result = distSRA
else
    x = x*10 %scale from [0,1] back to distance in meters
    result = squeeze(x(1,:,:));
end
if data_type == 2
    result = imresize(int32((result).*100),[180,240],'bilinear');
    result = double(result)./100;
end

mkdir(['~/ROS/DeepEnd2End/DeepToF_release_0.1/src/GAN/results/' mydata]);
imwrite(result, sprintf('~/ROS/DeepEnd2End/DeepToF_release_0.1/src/GAN/results/%s/this_depth.png',mydata),'Bitdepth',16);
%imagesc(squeeze(x(1,:,:))) %display tofnet output
%error = gt.-result;

figure(1);
suptitle(sprintf('%s-%d depth compare',mydata,take));
subplot(121); imagesc(gt); axis image; colorbar; title('gt');
subplot(122); imagesc(result); axis image; colorbar; title('result');
%subplot(133); imagesc(error); axis image; colorbar; title('error');

%%%%%%%%% visualize point clouds
%need sensor param
if data_type == 1
pointCloud_gt = depthToPointCloud(gt);
pointCloud_this = depthToPointCloud(result);
figure(2);
subplot(121); pcshow(reshape(pointCloud_gt,320*240,3)); title('point cloud of gt depth');
subplot(122); pcshow(reshape(pointCloud_this,320*240,3)); title('point cloud of this depth');

elseif data_type == 2
    pointCloud_gt = my_depthToPointCloud(gt,'no_calib');
    pointCloud_this = my_depthToPointCloud(result,'no_calib');
    figure(2);
    subplot(121); pcshow(reshape(pointCloud_gt,240*180,3)); title('point cloud of gt depth');
    subplot(122); pcshow(reshape(pointCloud_this,240*180,3)); title('point cloud of this depth');
end

