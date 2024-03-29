clc;clear;close all;
freqs = [45180000,37650000];
width = 240;
height = 180;
type = 'fake_B'; % fake_B,real_A,real_B

%change this to adapt your data
config = '0429';
GAN_path = "~/deep_tof/src/GAN" %change your gan path
test_dir = sprintf('%s/datasets/royal_itof_test/test_0429_1',GAN_path)   %change your test path   
result_dir = sprintf('%s/results/new_gan_128/latest_test_0429_1/images/%s/',GAN_path,type) %change your result path      
%end


resultList = dir(result_dir);
n=length(resultList);
if n==0
    sprintf('resultList file name is zero, may have error result_dir')
    return 
end

for i=3:n
    data_info = strsplit(resultList(i).name,'.')
    data_info = strsplit(data_info{1,1},'_pix2pix_test_real_')
    bag_name = data_info{1,1};
    time_stamp = data_info{1,2};
    file_name = sprintf('%s_pix2pix_test_real_%s.mat',bag_name,time_stamp);
         
    load(sprintf('%s/%s',result_dir,file_name))
    x = x*10;
    depth_gan_model = squeeze(x(1,:,:));
    depth_gan_model = imresize(int32((depth_gan_model).*100),[180,240],'bilinear');
    depth_gan_model = double(depth_gan_model)./100;
    
    
    load(sprintf('%s/%s',test_dir,file_name))
    depth_gt = permute(im_pair(1,:,end/2+1:end),[2,3,1]) * 10;

    Imat1 = permute(im_pair(1,:,1:end/2),[2,3,1]);
    Qmat1 = permute(im_pair(2,:,1:end/2),[2,3,1]);
    Imat2 = permute(im_pair(3,:,1:end/2),[2,3,1]);
    Qmat2 = permute(im_pair(4,:,1:end/2),[2,3,1]);
    Amp = permute(im_pair(5,:,1:end/2),[2,3,1]);


    max_depth = max([depth_gt(:); depth_gan_model(:)]);
    min_depth = min([depth_gt(:); depth_gan_model(:)]);

    figure(1);
    suptitle('raw data');
    subplot(321); imagesc(depth_gt); axis image; colorbar; title('gt');
    subplot(322); imagesc(Amp); axis image; colorbar; title('Amp');
    subplot(323); imagesc(Imat1); axis image; colorbar; title('Imat1');
    subplot(324); imagesc(Qmat1); axis image; colorbar; title('Qmat1');
    subplot(325); imagesc(Imat2); axis image; colorbar; title('Imat2');
    subplot(326); imagesc(Qmat2); axis image; colorbar; title('Qmat2');


    figure(2);
    suptitle(sprintf('%s %s depth compare',bag_name,time_stamp));
    subplot(141); imagesc(Amp); axis image; colorbar; title('Amp');
    subplot(142); imagesc(Imat1); axis image; colorbar; title('Imat1');
    % show gt depth 
    subplot(143); imagesc(depth_gt); axis image; caxis([min_depth,max_depth]); colorbar; title('depth (gt)');
    % show the learned depth
    subplot(144); imagesc(depth_gan_model); axis image; caxis([min_depth,max_depth]); colorbar; title('depth (GAN)');

    
    figure(3);
    pointCloud_gt = my_depthToPointCloud(depth_gt,config);
    pointCloud_gan = my_depthToPointCloud(depth_gan_model,config);
    subplot(121); pcshow(reshape(pointCloud_gt,240*180,3)); title('cloud (gt)');
    subplot(122); pcshow(reshape(pointCloud_gan,240*180,3)); title('cloud (GAN)');
    
    pause(5);
end


