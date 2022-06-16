clc; clear; close all
data_type = 1;
run_single = 0;
is_visual = 0;
is_saving = 1;

target_width = 240; %320 240
target_height = 180; %240 180

if data_type == 1
  datasets_dir = '~/ROS/DeepEnd2End/DeepToF_release_0.1/src/GAN/datasets/mpi_correction_corrnormamp2depth_albedoaug_zpass/train'
  data_name ='000980.mat'
else
  datasets_dir = '/media/lixin/7A255A482B58BC84/lx/0428/deep_tof_datasets/train_data/train'
  data_name ='_2022-04-28-16-41-10_pix2pix_test_real_1.mat'
end

if is_saving
    out_path = sprintf('%s/../resize_train',datasets_dir)
    mkdir([out_path]);
end
datasets_list = dir(datasets_dir);
n = length(datasets_list);
for i=3:n
    if run_single == 0
        data_name = datasets_list(i).name
    end 
    load(sprintf('%s/%s',datasets_dir,data_name))
    dist_gt = permute(im_pair(1,:,end/2+1:end),[2,3,1]);
    dist_gt = imresize(single(dist_gt),[target_height,target_width],'bilinear');
    Imat1 = permute(im_pair(1,:,1:end/2),[2,3,1]);
    Imat1 = imresize(single(Imat1),[target_height,target_width],'bilinear');
    Qmat1 = permute(im_pair(2,:,1:end/2),[2,3,1]);
    Qmat1 = imresize(single(Qmat1),[target_height,target_width],'bilinear');    
    Imat2 = permute(im_pair(3,:,1:end/2),[2,3,1]);
    Imat2 = imresize(single(Imat2),[target_height,target_width],'bilinear');    
    Qmat2 = permute(im_pair(4,:,1:end/2),[2,3,1]);
    Qmat2 = imresize(single(Qmat2),[target_height,target_width],'bilinear');    
    Amp = permute(im_pair(5,:,1:end/2),[2,3,1]);
    Amp = imresize(single(Amp),[target_height,target_width],'bilinear');     
    
    corr = cat(1, permute(Imat1,[3,1,2]), permute(Qmat1,[3,1,2]), permute(Imat2,[3,1,2]), permute(Qmat2,[3,1,2]), permute(Amp,[3,1,2]));
    depth = cat(1, permute(dist_gt,[3,1,2]), permute(dist_gt,[3,1,2]), permute(dist_gt,[3,1,2]), permute(dist_gt,[3,1,2]), permute(dist_gt,[3,1,2]));
    im_pair = cat(3,corr,depth);
    if is_saving
        save(sprintf('%s/%s',out_path,data_name),'im_pair');
    end
    if is_visual == 0
        continue
    end
    
    dist_gt = dist_gt * 10;


    figure(1);
    suptitle(sprintf('%s',data_name));
    subplot(161); imagesc(dist_gt); axis image; colorbar; title('gt');
    subplot(162); imagesc(Amp); axis image; colorbar; title('Amp');
    subplot(163); imagesc(Imat1); axis image; colorbar; title('Imat1');
    subplot(164); imagesc(Qmat1); axis image; colorbar; title('Qmat1');
    subplot(165); imagesc(Imat2); axis image; colorbar; title('Imat2');
    subplot(166); imagesc(Qmat2); axis image; colorbar; title('Qmat2');
    set(gcf,'Position',[200 200 3000 400]);
    % 
    %%%%%%%%% visualize point clouds
    %need sensor param
    pointCloud_gt = my_depthToPointCloud(dist_gt,'0429');
    figure(2);
    pcshow(reshape(pointCloud_gt,target_width*target_height,3)); title('point cloud of gt depth');

    if run_single
        break;
    end
end