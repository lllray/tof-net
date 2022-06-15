clc;clear;close all;
freqs = [45180000,37650000];
width = 240;
height = 180;
% 
% model_name_gan = 'stereo0428_my_resnet_lite_my_imageGAN_128'; % my_resnet_lite_my_imageGAN_128_pretrained,22-04-20-num4,my_resnet_lite_my_imageGAN_128
% data_name = "latest_test_stereo_little_0429";
% show_sra = 0;
% show_base = 0;
% show_source = 1;
% model_name_base = '22-04-21-3_my_resnet_lite_my_imageGAN_128';
% type = 'fake_B'; % fake_B,real_A,real_B,sra
% mydata = '/media/lixin/7A255A482B58BC84/lx/0429/little_test/train';
% source_path = '/media/lixin/7A255A482B58BC84/lx/0429/little_test/phase_calibrated_norm2amp_rebuttal_mean1';
% bag_name = '_2022-04-29-15-30-30';
% take = 50;
% file_name = sprintf('%s_pix2pix_test_real_%d.mat',bag_name,take);
% source_name = sprintf('%s_source_depth_%d.mat',bag_name,take); 
% config = '0429';
config = '0429';
model_name_gan = 'stereo0428_my_resnet_lite_my_imageGAN_128'; % my_resnet_lite_my_imageGAN_128_pretrained,22-04-20-num4,my_resnet_lite_my_imageGAN_128
data_name = "latest_test_stereo_little_04292";
show_sra = 0;
show_base = 0;
show_source = 1;
show_left_image = 1;
save_depth_compare_image = 1;

model_name_base = '22-04-21-3_my_resnet_lite_my_imageGAN_128';
type = 'fake_B'; % fake_B,real_A,real_B,sra
mydata = '/media/lixin/7A255A482B58BC84/lx/0429/deep_tof_datasets/impair_mean1';
source_path = '/media/lixin/7A255A482B58BC84/lx/0429/deep_tof_datasets/phase_calibrated_norm2amp_rebuttal_mean1';

bag_name = '_2022-04-29-14-31-51';
take = 10;
file_name = sprintf('%s_pix2pix_test_real_%d.mat',bag_name,take);
source_name = sprintf('%s_source_depth_%d.mat',bag_name,take); 
left_image_path = sprintf('%s/../%s/left',mydata,bag_name); 


% model_name_gan = 'stereo0428_my_resnet_lite_my_imageGAN_128'; % my_resnet_lite_my_imageGAN_128_pretrained,22-04-20-num4,my_resnet_lite_my_imageGAN_128
% data_name = "latest_test_stereo_0428";
% show_sra = 0;
% show_base = 0;
% show_source = 1;
% model_name_base = '22-04-21-3_my_resnet_lite_my_imageGAN_128';
% type = 'fake_B'; % fake_B,real_A,real_B,sra
% mydata = '/media/lixin/7A255A482B58BC84/lx/0428/deep_tof_datasets/train_data/train';
% source_path = '/media/lixin/7A255A482B58BC84/lx/0428/deep_tof_datasets/train_data/phase_calibrated_norm2amp_rebuttal_mean1';
% bag_name = '_2022-04-28-16-48-41';
% take = 109;
% file_name = sprintf('%s_pix2pix_test_real_%d.mat',bag_name,take);
% source_name = sprintf('%s_source_depth_%d.mat',bag_name,take); 
% config = '0428';
if show_left_image
    fileList = dir(left_image_path);
    sprintf('%s/%s',left_image_path,fileList(take+2).name)
    left_image = imread(sprintf('%s/%s',left_image_path,fileList(take+2).name));
end

depth_figure_count = 140 + show_sra * 10 + show_base * 10 + show_source*10 + show_left_image*10;
cloud_figure_count = 120 + show_sra * 10 + show_base * 10 + show_source*10;
if show_sra
load(sprintf('~/ROS/DeepEnd2End/DeepToF_release_0.1/src/GAN/results/%s/%s/images/sra/%s',model_name_gan,data_name,file_name)) %results are stored as .mat files under [results_folder]/fake_B
depth_sra = distSRA;
depth_sra = medfilt2(depth_sra,[3,3]);
depth_sra = imresize(int32((depth_sra).*100),[180,240],'bilinear');
depth_sra = double(depth_sra)./100;
end

if show_base
load(sprintf('~/ROS/DeepEnd2End/DeepToF_release_0.1/src/GAN/results/%s/%s/images/%s/%s',model_name_base,data_name,type,file_name)) %results are stored as .mat files under [results_folder]/fake_B
x = x*10;
depth_l1_model = squeeze(x(1,:,:));
depth_l1_model = imresize(int32((depth_l1_model).*100),[180,240],'bilinear');
depth_l1_model = double(depth_l1_model)./100;
end

load(sprintf('~/ROS/DeepEnd2End/DeepToF_release_0.1/src/GAN/results/%s/%s/images/%s/%s',model_name_gan,data_name,type,file_name)) %results are stored as .mat files under [results_folder]/fake_B
x = x*10;
depth_gan_model = squeeze(x(1,:,:));
depth_gan_model = imresize(int32((depth_gan_model).*100),[180,240],'bilinear');
depth_gan_model = double(depth_gan_model)./100;

maxd = 10;
load(sprintf('%s/%s',mydata,file_name));

depth_gt = permute(im_pair(1,:,end/2+1:end),[2,3,1]) * 10;

load(sprintf('%s/%s',source_path,source_name));

depth_source = s_depth;

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


max_depth = max([depth_gt(:); depth_gan_model(:)]);
min_depth = min([depth_gt(:); depth_gan_model(:)]);

figure(2);
suptitle(sprintf('%s %d depth compare',bag_name,take));
if show_left_image
    subplot(depth_figure_count + show_left_image); imagesc(left_image); axis image; title('image');
end
subplot(depth_figure_count + 1 + show_left_image); imagesc(Amp); axis image; colorbar; title('Amp');
subplot(depth_figure_count + 2 + show_left_image); imagesc(Imat1); axis image; colorbar; title('Imat1');
% show depth from tintin output
subplot(depth_figure_count +3 + show_left_image); imagesc(depth_gt); axis image; caxis([min_depth,max_depth]); colorbar; title('depth (gt)');
% show depth from phase unwrapping
if show_source
subplot(depth_figure_count +4 + show_left_image); imagesc(depth_source); axis image; caxis([min_depth,max_depth]); colorbar; title('depth (royal)');
end
if show_sra
subplot(depth_figure_count +4+show_source + show_left_image); imagesc(depth_sra); axis image; caxis([min_depth,max_depth]); colorbar; title('depth (sra)');
end
% show the learned depth
if show_base
subplot(depth_figure_count +4 +show_source+ show_sra + show_left_image); imagesc(depth_l1_model); axis image; caxis([min_depth,max_depth]); colorbar; title('depth (L1)');
end
% show the learned depth
subplot(depth_figure_count +4 +show_source+ show_sra + show_base + show_left_image); imagesc(depth_gan_model); axis image; caxis([min_depth,max_depth]); colorbar; title('depth (GAN)');
set(gcf,'Position',[200 200 3600 400]);
if save_depth_compare_image
    saveas(gcf, sprintf('%s_%d.jpg',bag_name,take));
end
% compare scanlines
figure(3);
suptitle('compare depth');
plot(depth_gt(end/2,:)); hold on; 
if show_sra
plot(depth_sra(end/2,:)); hold on;
end
if show_source
plot(depth_source(end/2,:)); hold on;
end
if show_base
plot(depth_l1_model(end/2,:)); hold on
end
plot(depth_gan_model(end/2,:)); hold off

if show_sra && show_base
legend('depth (gt)', 'depth (sra)', 'depth (L1)','depth (GAN)');
elseif show_source
    legend('depth (gt)', 'depth (royal)','depth (GAN)');
elseif show_sra
    legend('depth (gt)', 'depth (sra)','depth (GAN)');
elseif show_base
    legend('depth (gt)', 'depth (L1)','depth (GAN)');
else
    legend('depth (gt)','depth (GAN)');
end
title('scanline (middle row)');


pointCloud_gt = my_depthToPointCloud(depth_gt,config);
if show_sra
pointCloud_sra = my_depthToPointCloud(depth_sra,config);
end
if show_base
pointCloud_l1 = my_depthToPointCloud(depth_l1_model,config);
end
pointCloud_gan = my_depthToPointCloud(depth_gan_model,config);
figure(4);
subplot(cloud_figure_count + 1); pcshow(reshape(pointCloud_gt,240*180,3)); title('cloud (gt)');
if show_source
pointCloud_source = my_depthToPointCloud(depth_source,config);    
subplot(cloud_figure_count + 2); pcshow(reshape(pointCloud_source,240*180,3)); title('cloud (royal)');
end
if show_sra
subplot(cloud_figure_count + 2 + show_source); pcshow(reshape(pointCloud_sra,240*180,3)); title('cloud (sra)');
end
if show_base
subplot(cloud_figure_count + 2 + show_source + show_sra); pcshow(reshape(pointCloud_l1,240*180,3)); title('cloud (L1)');
end
subplot(cloud_figure_count + 2+ show_source + show_sra + show_base); pcshow(reshape(pointCloud_gan,240*180,3)); title('cloud (GEN)');

