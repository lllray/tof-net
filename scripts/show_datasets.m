clc; clear; close all
data_type = 2;
run_single = 0;
save_image = 1;
step = 10;

if data_type == 1
  datasets_dir = '~/ROS/DeepEnd2End/DeepToF_release_0.1/src/GAN/datasets/mpi_correction_corrnormamp2depth_albedoaug_zpass/train'
  data_name ='000980.mat'
else
  %datasets_dir = '/media/lixin/7A255A482B58BC84/lx/0428/deep_tof_datasets/train_data/train'
  datasets_dir = '/media/lixin/7A255A482B58BC84/lx/0428/deep_tof_datasets/train_data/train'
  %data_name ='_2022-04-28-16-41-10_pix2pix_test_real_1.mat'
  data_name ='000980.mat'
end

if save_image
    out_path = sprintf('%s/../visual',datasets_dir)
    mkdir([out_path]);
end


datasets_list = dir(datasets_dir);
n = length(datasets_list);
for i=3:step:n
    if run_single == 0
        data_name = datasets_list(i).name
    end 
    load(sprintf('%s/%s',datasets_dir,data_name))
    dist_gt = permute(im_pair(1,:,end/2+1:end),[2,3,1]) * 10;

    Imat1 = permute(im_pair(1,:,1:end/2),[2,3,1]);
    Qmat1 = permute(im_pair(2,:,1:end/2),[2,3,1]);
    Imat2 = permute(im_pair(3,:,1:end/2),[2,3,1]);
    Qmat2 = permute(im_pair(4,:,1:end/2),[2,3,1]);
    Amp = permute(im_pair(5,:,1:end/2),[2,3,1]);


    figure(1);
    suptitle(sprintf('%s',data_name));
    subplot(161); imagesc(dist_gt); axis image; colorbar; title('gt');
    subplot(162); imagesc(Amp); axis image; colorbar; title('Amp');
    subplot(163); imagesc(Imat1); axis image; colorbar; title('Imat1');
    subplot(164); imagesc(Qmat1); axis image; colorbar; title('Qmat1');
    subplot(165); imagesc(Imat2); axis image; colorbar; title('Imat2');
    subplot(166); imagesc(Qmat2); axis image; colorbar; title('Qmat2');
    set(gcf,'Position',[200 200 3000 400]);
    if save_image
        saveas(gcf, sprintf('%s/%s.jpg',out_path,data_name));
    end
    % 
    %%%%%%%%% visualize point clouds
    %need sensor param
    if save_image == 0
        if data_type == 1
            pointCloud_gt = depthToPointCloud(dist_gt);
            figure(2);
            pcshow(reshape(pointCloud_gt,320*240,3)); title('point cloud of gt depth');
        elseif data_type == 2
            pointCloud_gt = my_depthToPointCloud(dist_gt,'0429');
            figure(2);
            pcshow(reshape(pointCloud_gt,240*180,3)); title('point cloud of gt depth');
        end
    end
    if run_single
        break;
    elseif save_image == 0
        pause(5)
    end
end