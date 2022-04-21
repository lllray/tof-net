clc;clear;close all;
freqs = [45180000,37650000];
width = 240;
height = 180;
model_name_base = '22-04-21-3-ng_my_resnet_lite_my_imageGAN_128';
model_name_gan = '22-04-21-3_my_resnet_lite_my_imageGAN_128'; % my_resnet_lite_my_imageGAN_128_pretrained,22-04-20-num4,my_resnet_lite_my_imageGAN_128
data_name = "latest_test_real_all_calib_my_room";
type = 'fake_B'; % fake_B,real_A,real_B,sra
mydata = 'dataset-0421-3';
take = 34;

load(sprintf('~/ROS/DeepEnd2End/DeepToF_release_0.1/src/GAN/results/%s/%s/images/sra/%s_pix2pix_test_real_%d.mat',model_name_gan,data_name,mydata,take)) %results are stored as .mat files under [results_folder]/fake_B
depth_sra = distSRA;
depth_sra = medfilt2(depth_sra,[3,3]);

load(sprintf('~/ROS/DeepEnd2End/DeepToF_release_0.1/src/GAN/results/%s/%s/images/%s/%s_pix2pix_test_real_%d.mat',model_name_base,data_name,type,mydata,take)) %results are stored as .mat files under [results_folder]/fake_B
x = x*10;
depth_l1_model = squeeze(x(1,:,:));
load(sprintf('~/ROS/DeepEnd2End/DeepToF_release_0.1/src/GAN/results/%s/%s/images/%s/%s_pix2pix_test_real_%d.mat',model_name_gan,data_name,type,mydata,take)) %results are stored as .mat files under [results_folder]/fake_B
x = x*10;
depth_gan_model = squeeze(x(1,:,:));

maxd = 10;
load(sprintf('~/bag/tintin_EE367/my_data/%s/train/%s_pix2pix_test_real_%d.mat',mydata,mydata,take));

depth_gt = permute(im_pair(1,:,end/2+1:end),[2,3,1]) * 10;

Imat1 = permute(im_pair(1,:,1:end/2),[2,3,1]);
Qmat1 = permute(im_pair(2,:,1:end/2),[2,3,1]);
Imat2 = permute(im_pair(3,:,1:end/2),[2,3,1]);
Qmat2 = permute(im_pair(4,:,1:end/2),[2,3,1]);
Amp = permute(im_pair(5,:,1:end/2),[2,3,1]);


figure(1);
suptitle('raw data');
subplot(321); imagesc(depth_gt); axis image; colorbar; title('gt');
subplot(322); imagesc(Amp); axis image; colorbar; title('Amp');
subplot(323); imagesc(Imat1); axis image; colorbar; title('Imat1');
subplot(324); imagesc(Qmat1); axis image; colorbar; title('Qmat1');
subplot(325); imagesc(Imat2); axis image; colorbar; title('Imat2');
subplot(326); imagesc(Qmat2); axis image; colorbar; title('Qmat2');


max_depth = max([depth_gt(:); depth_sra(:); depth_l1_model(:); depth_gan_model(:)]);
min_depth = min([depth_gt(:); depth_sra(:); depth_l1_model(:); depth_gan_model(:)]);

figure(2);
suptitle(sprintf('%s-%d depth compare',mydata,take));
subplot(161); imagesc(Amp); axis image; colorbar; title('Amp');
subplot(162); imagesc(Imat1); axis image; colorbar; title('Imat1');
% show depth from tintin output
subplot(163); imagesc(depth_gt); axis image; caxis([min_depth,max_depth]); colorbar; title('depth (gt)');
% show depth from phase unwrapping
subplot(164); imagesc(depth_sra); axis image; caxis([min_depth,max_depth]); colorbar; title('depth (sra)');
% show the learned depth
subplot(165); imagesc(depth_l1_model); axis image; caxis([min_depth,max_depth]); colorbar; title('depth (L1)');
% show the learned depth
subplot(166); imagesc(depth_gan_model); axis image; caxis([min_depth,max_depth]); colorbar; title('depth (GAN)');

% compare scanlines
figure(3);
suptitle('compare depth');
plot(depth_gt(end/2,:)); hold on; 
plot(depth_sra(end/2,:)); hold on;
plot(depth_l1_model(end/2,:)); hold on
plot(depth_gan_model(end/2,:)); hold off
legend('depth (gt)', 'depth (sra)', 'depth (L1)','depth (GAN)');
title('scanline (middle row)');


pointCloud_gt = my_depthToPointCloud(depth_gt,'no_calib');
pointCloud_sra = my_depthToPointCloud(depth_sra,'no_calib');
pointCloud_l1 = my_depthToPointCloud(depth_l1_model,'no_calib');
pointCloud_gan = my_depthToPointCloud(depth_gan_model,'no_calib');
figure(4);
subplot(141); pcshow(reshape(pointCloud_gt,240*180,3)); title('cloud (gt)');
subplot(142); pcshow(reshape(pointCloud_sra,240*180,3)); title('cloud (sra)');
subplot(143); pcshow(reshape(pointCloud_l1,240*180,3)); title('cloud (L1)');
subplot(144); pcshow(reshape(pointCloud_gan,240*180,3)); title('cloud (GEN)');

