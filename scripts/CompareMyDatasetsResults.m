clc;clear;close all;
freqs = [45180000,37650000];
width = 240;
height = 180;

config = '0429';
model_name_gan = 'stereo_data_inverse_resnet_lite_my_imageGAN_128'; % my_resnet_lite_my_imageGAN_128_pretrained,22-04-20-num4,my_resnet_lite_my_imageGAN_128
data_name = "latest_train";
show_royal = 1;
show_left_image = 0;
save_depth_compare_image = 0;
run_single = 0;
type = 'fake_B'; % fake_B,real_A,real_B,sra
mydata = '/media/lixin/7A255A482B58BC84/lx/itof_datasets/test_data/train';
source_path = '/media/lixin/7A255A482B58BC84/lx/itof_datasets/test_data/depth';
out_path = sprintf('%s/../result',mydata)
mkdir([out_path]);

bag_name = '_2022-04-29-14-31-51';
time_stamp = '1651213921040147';

    sprintf('%s/../%s/%s/images/%s/',mydata,model_name_gan,data_name,type)
    testList = dir(sprintf('%s/../%s/%s/images/%s/',mydata,model_name_gan,data_name,type));
    n=length(testList);
    for i=3:n
        if run_single == 0
            data_info = strsplit(testList(i).name,'.')
            data_info = strsplit(data_info{1,1},'_pix2pix_test_real_')
            bag_name = data_info{1,1};
            time_stamp = data_info{1,2};
        end
        file_name = sprintf('%s_pix2pix_test_real_%s.mat',bag_name,time_stamp);
        source_name = sprintf('%s_source_depth_%s.mat',bag_name,time_stamp); 
        left_image_name = sprintf('%s/../%s/left/%s.jpg',mydata,bag_name,time_stamp);
         
        if show_left_image
            left_image = imread(left_image_name);
        end

        depth_figure_count = 140  + show_royal*10 + show_left_image*10;
        cloud_figure_count = 120  + show_royal*10;

        load(sprintf('%s/../%s/%s/images/%s/%s',mydata,model_name_gan,data_name,type,file_name)) %results are stored as .mat files under [results_folder]/fake_B
        x = 1/x - 1;
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


        max_depth = max([depth_gt(:); depth_gan_model(:)]);
        min_depth = min([depth_gt(:); depth_gan_model(:)]);
        if save_depth_compare_image == 0
            figure(1);
            suptitle('raw data');
            subplot(321); imagesc(depth_gt); axis image; colorbar; title('gt');
            subplot(322); imagesc(Amp); axis image; colorbar; title('Amp');
            subplot(323); imagesc(Imat1); axis image; colorbar; title('Imat1');
            subplot(324); imagesc(Qmat1); axis image; colorbar; title('Qmat1');
            subplot(325); imagesc(Imat2); axis image; colorbar; title('Imat2');
            subplot(326); imagesc(Qmat2); axis image; colorbar; title('Qmat2');
        end

        figure(2);
        suptitle(sprintf('%s %s depth compare',bag_name,time_stamp));
        if show_left_image
            subplot(depth_figure_count + show_left_image); imagesc(left_image); axis image; title('image');
        end
        subplot(depth_figure_count + 1 + show_left_image); imagesc(Amp); axis image; colorbar; title('Amp');
        subplot(depth_figure_count + 2 + show_left_image); imagesc(Imat1); axis image; colorbar; title('Imat1');
        % show depth from tintin output
        subplot(depth_figure_count +3 + show_left_image); imagesc(depth_gt); axis image; caxis([min_depth,max_depth]); colorbar; title('depth (gt)');
        % show depth from phase unwrapping
        if show_royal
        subplot(depth_figure_count +4 + show_left_image); imagesc(depth_source); axis image; caxis([min_depth,max_depth]); colorbar; title('depth (royal)');
        end
        % show the learned depth
        subplot(depth_figure_count +4 +show_royal + show_left_image); imagesc(depth_gan_model); axis image; caxis([min_depth,max_depth]); colorbar; title('depth (GAN)');
        set(gcf,'Position',[200 200 3600 400]);
        if save_depth_compare_image
            saveas(gcf, sprintf('%s/%s_%s.jpg',out_path,bag_name,time_stamp));
        end
        % compare scanlines
        if save_depth_compare_image == 0
            figure(3);
            suptitle('compare depth');
            plot(depth_gt(end/2,:)); hold on; 
            if show_royal
            plot(depth_source(end/2,:)); hold on;
            end
            plot(depth_gan_model(end/2,:)); hold off

            if show_royal
                legend('depth (gt)', 'depth (royal)','depth (GAN)');
            else
                legend('depth (gt)','depth (GAN)');
            end
            title('scanline (middle row)');
        end

            pointCloud_gt = my_depthToPointCloud(depth_gt,config);
            pointCloud_gan = my_depthToPointCloud(depth_gan_model,config);
            figure(4);
            subplot(cloud_figure_count + 1); pcshow(reshape(pointCloud_gt,240*180,3)); title('cloud (gt)');
            if show_royal  
            pointCloud_source = my_depthToPointCloud(depth_source,config);    
            subplot(cloud_figure_count + 2); pcshow(reshape(pointCloud_source,240*180,3)); title('cloud (royal)');
            end
            subplot(cloud_figure_count + 2+ show_royal); pcshow(reshape(pointCloud_gan,240*180,3)); title('cloud (GEN)');
            set(gcf,'Position',[200 200 1800 400]);
            if save_depth_compare_image
                saveas(gcf, sprintf('%s/%s_%scloud.jpg',out_path,bag_name,time_stamp));
            end
        
        if run_single
            break;
        elseif save_depth_compare_image == 0
             pause(5);
        end
    end


