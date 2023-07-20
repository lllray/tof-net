% % given tintin depth and raw measurements
% % exact I, Q and depth from the binary
% % calculate depth_pu using phase unwrapping
% % prepare correct data format for network prediction (my ed and pix2pix)

clc; clear; close all

max_depth_norm = 10; % max depth for normalization to [0,1]
max_depth_vis = 10; % max depth for visualization
is_visualizing = false;
is_saving_debug = false;
is_saving_release = true;
normalization = 2;
normalization_amp = 0;%6000
catamp = 1; % if concat amp image with imA
calib_phase_offset = 0; 
navg = 1; % number of measurements to average
datasets_folder = '/media/lixin/7A255A482B58BC84/lx/0429/deep_tof_datasets/';
depth_folder = sprintf('train_data_1/depth');
data_folder = sprintf('train_data_1/train');
config = '0429';
data_from_datasets = 1;
load_left_image = 1
all_min_amp = 99999;
all_max_amp = 0;
mkdir([datasets_folder '/' depth_folder]);
mkdir([datasets_folder '/' data_folder]);

    %-----  my data  ---------
    if data_from_datasets == 1
        fileList = dir(datasets_folder);
        n=length(fileList);
        date = {};
        for i=3:n
            date = [date fileList(i).name];
        end
    else
        date = {'_2022-04-28-16-51-25'};
    end
    freqs = [45180000,37650000];
    takes = 0;
    width = 240;
    height = 180;

    for idate = 1:numel(date)
        folder = [datasets_folder date{idate}]
        itakes = 1;
            
            %%%%%%%%% process depth
            fnd = sprintf('%s/stereo_depth_in_itof.txt',folder);
            if ~exist(fnd,'file')
                printf("not find depth data")
               continue;
            end
            depth_data = importdata(fnd).';
            
            fnsd = sprintf('%s/itof_output_depth.txt',folder);
            if ~exist(fnsd,'file')
                printf("not find source depth data")
               continue;
            end
            source_depth_data = importdata(fnsd).';
            
            fnsgmd = sprintf('%s/sgm_depth_in_itof.txt',folder);
            if ~exist(fnsd,'file')
                printf("not find sgm depth data")
               continue;
            end
            sgm_depth_data = importdata(fnsgmd).';
            
            raw_data = [];
            for ifreqs = 1:numel(freqs)
                fnc = sprintf('%s/itof_output_%d.txt',folder,freqs(ifreqs));
                 if ~exist(fnc,'file')
                     printf("not find raw data")
                      is_corr_good = false;
                      return;
                 end
                 raw_data = cat(3,raw_data,importdata(fnc).');
            end
            [d_r,d_c] = size(depth_data)
            [sd_r,sd_c] = size(source_depth_data);
            [sgmd_r,sgmd_c] = size(sgm_depth_data);
            [raw_r,raw_c,raw_n] = size(raw_data);
            if raw_c ~= d_c
                printf("depth data num is not eual raw data num")
            end
            timestamp_list = source_depth_data(1,:);
            sgm_timestamp_list = sgm_depth_data(1,:);
            for i = 1:d_c
                depth = depth_data(:,i);
                timestamp = depth(1);
                depth = depth(4:3+height*width);
                depth = reshape(depth,width,height)';
                depth(depth<=0.0) = nan; % is importan to set nan
                depth(depth>=max_depth_norm) = max_depth_norm; % is importan to set nan
                find = 0;
                for j = 1:sd_c
                    if timestamp == timestamp_list(j)
                        find = 1;
                        break;
                    end
                end 
                if find == 0
                    continue
                end
                find = 0;
                for k = 1:sgmd_c
                    if timestamp == sgm_timestamp_list(k)
                        find = 1;
                        break;
                    end
                end
                if find == 0
                    continue
                end
                s_depth = source_depth_data(:,j);
                s_depth = s_depth(4:3+height*width);
                s_depth = reshape(s_depth,width,height)';
                s_depth(s_depth<=0.0) = nan; % is importan to set nan
                s_depth(s_depth>=max_depth_norm) = max_depth_norm; % is importan to set nan
                sgm_depth = sgm_depth_data(:,k);
                sgm_depth = sgm_depth(4:3+height*width);
                sgm_depth = reshape(sgm_depth,width,height)';
                sgm_depth(sgm_depth<=0.0) = nan; % is importan to set nan
                sgm_depth(sgm_depth>=max_depth_norm) = max_depth_norm; % is importan to set nan
                
                left_image_name = sprintf('%s/left/%ld.jpg',folder,timestamp);
                if load_left_image
                    left_image = imread(left_image_name);
                end
                if is_visualizing
                figure(1);
                suptitle(fnd);
                subplot(221); imagesc(depth); axis image; colorbar; title('depth');
                subplot(222); imagesc(s_depth); axis image; colorbar; title('source depth');
                subplot(223); imagesc(sgm_depth); axis image; colorbar; title('sgm depth');
                    if load_left_image
                    subplot(224); imagesc(left_image); axis image; title('color_image');
                    end
                end
                
                if is_saving_release
                    save(sprintf('%s/%s/%s_depth_%ld.mat',datasets_folder,depth_folder,date{idate},timestamp),'depth');
                    imwrite(depth/max_depth_vis, sprintf('%s/%s/%s_depth_%ld.png',datasets_folder,depth_folder,date{idate},timestamp),'Bitdepth',16);
                end
                
                h = zeros(numel(freqs)*2,height,width);
                for ifreqs = 1:numel(freqs)
                    raw = raw_data(:,j,ifreqs);
                        phase_Q = raw(4:3+width*height);
                        phase_I = raw(4+width*height:end);
                        I_Mat = reshape(phase_I,width,height)';
                        Q_Mat = reshape(phase_Q,width,height)';
                        if calib_phase_offset
                            load ~/bag/tintin_EE367/src/CalibratePhaseOffsetReal;
                            offsets(1) = -pi/2;
                            offsets(2) = -pi/2;
                            ov = cos(offsets(ifreqs)) + 1i*sin(offsets(ifreqs)); % offset vector, assumes 40, 70MHz
                            ov = ov./abs(ov); % norm 1
                            tmp = I_Mat+1i*Q_Mat;
                            tmp = tmp./ov;
                            I_Mat = real(tmp);
                            Q_Mat = imag(tmp);
                        end
                        Amp = abs(I_Mat+1i*Q_Mat);
                        Phase = angle(I_Mat+1i*Q_Mat);
                        if is_visualizing
                            figure(ifreqs+1); 
                            suptitle(sprintf('%d',floor(freqs(ifreqs))));
                            subplot(221); imagesc(I_Mat); axis image; colorbar; title('I');
                            subplot(222); imagesc(Q_Mat); axis image; colorbar; title('Q');
                            subplot(223); imagesc(Amp); axis image; colorbar; title('amp');
                            subplot(224); imagesc(Phase); axis image; colorbar; title('phase');
                        end
                        if is_saving_debug
                            save(sprintf('%s/%s/%s_amp_%ld_%d.mat',folder,depth_folder,date{idate},timestamp,freqs(ifreqs)),'Amp');
                            save(sprintf('%s/%s/%s_phase_%ld_%d.mat',folder,depth_folder,date{idate},timestamp,freqs(ifreqs)),'Phase');
                        end
                        h(ifreqs,:,:) = I_Mat;
                        h(ifreqs+numel(freqs),:,:) = Q_Mat;
                end

                %%%%%%%%% save corr for network prediction (my implementation of ed)
                corr_imgs = h;
                if is_saving_debug
                    save(sprintf('%s/%s/real_%s_all_%ld_corr_imgs.mat',folder,depth_folder,date{idate},timestamp),'corr_imgs');
                end

                %%%%%%%%% save image pair for pix2pix based GAN network
                % normalization (should not change the contrast by substracting minamp)
                nf = size(corr_imgs,1)/2;
                corr_imgs_abs = abs(corr_imgs(1:end/2,:,:) + 1i*corr_imgs(end/2+1:end,:,:));
        %         imagesc([squeeze(corr_imgs_abs(1,:,:)) squeeze(corr_imgs_abs(2,:,:))]); axis image; colorbar; pause;
                if normalization == 0 % just globally normalize to [0,1] for the whole stack
                    corr = corr_imgs;
                    maxamp = max(corr_imgs_abs(:));
                    corr = corr / maxamp;
                    corr = (corr + 1) / 2; % convert from [-1,1] to [0,1]
                elseif normalization == 1 % globally normalize each freq meas
                    corr = corr_imgs;
                    for ifreq = 1:nf*2
                        tmp = corr_imgs_abs(mod(ifreq-1,nf)+1,:,:);
                        maxamp = max(tmp(:));
                        corr(ifreq,:,:) = corr(ifreq,:,:)/maxamp;
                    end
                    corr = (corr + 1) / 2; % convert from [-1,1] to [0,1]
                elseif normalization == 2 % normalize at each pixel
                    corr = corr_imgs ./ cat(1, corr_imgs_abs, corr_imgs_abs);
                    corr = (corr + 1) / 2; % convert from [-1,1] to [0,1]
                end
                corr = corr([1,nf,nf+1,end],:,:); % choose some freq (if numel(freqs)>2)
                if catamp
                    if normalization_amp <= 0
                        tmp = corr_imgs_abs(1,:,:);
                        minamp = min(tmp(:));
                        maxamp = max(tmp(:));
                        all_min_amp = min(all_min_amp,minamp)
                        all_max_amp = max(all_max_amp,maxamp)
                        amp = tmp-minamp;
                        if maxamp > minamp
                            amp = amp / (maxamp-minamp);
                        end
                        corr = cat(1,corr,amp,amp);
                    else
                        tmp = corr_imgs_abs(1,:,:);
                        tmp(tmp>=normalization_amp) = normalization_amp;
                        amp = tmp / normalization_amp;
                        corr = cat(1,corr,amp,amp);
                    end
                end

                % normalize, assemble
                depth_pair = reshape(depth, 1, size(depth,1), size(depth,2));
                s_depth_pair = reshape(s_depth, 1, size(s_depth,1), size(s_depth,2));
                sgm_depth_pair = reshape(sgm_depth, 1, size(sgm_depth,1), size(sgm_depth,2));
                depth_pair = depth_pair / max_depth_norm; % convert to [0,1]
                s_depth_pair = s_depth_pair / max_depth_norm; % convert to [0,1]
                sgm_depth_pair = sgm_depth_pair / max_depth_norm;
                if load_left_image
                        left_image = imresize(left_image,[height,width],'bilinear');     
                        
                        left_image_pu_pair = im2double(permute(left_image,[3,1,2]));
                end
                if ~catamp
                    depth_pair = cat(1, depth_pair, s_depth_pair, depth_pair, depth_pair);
                elseif load_left_image
                    depth_pair = cat(1, depth_pair, s_depth_pair, left_image_pu_pair,sgm_depth_pair);
                else     
                    depth_pair = cat(1, depth_pair, s_depth_pair, depth_pair, depth_pair, depth_pair);
                end

                % combine for pix2pix aligned data loader
                im_pair = cat(3,corr,depth_pair);
                % split and save
                if is_saving_release
                    save(sprintf('%s/%s/%s_pix2pix_test_real_%ld',datasets_folder,data_folder,date{idate},timestamp),'im_pair');
                end
                
                
                %%%%%%%%% visualize point clouds
                %need sensor param
                if is_visualizing
                    pointCloud = my_depthToPointCloud(depth,config);
                    pointCloud_source = my_depthToPointCloud(s_depth,config);
                    
                    phase = angle(corr_imgs(1:end/2,:,:) + 1i*corr_imgs(end/2+1:end,:,:));
                    phase = phase([1,end],:,:);
                    freqsm = freqs;
                    lambda = 3e8./freqsm;
                    phase(phase<0) = 2*pi + phase(phase<0);
                    tic;
                    depth_phaseunwarp = PhaseImgs2Depths(freqsm, phase, 0:0.02:10);
                    pointCloud_pu = my_depthToPointCloud(depth_phaseunwarp,config);
                    
                    figure(numel(freqs)+2);
                    subplot(131); pcshow(reshape(pointCloud,width*height,3)); title('point cloud of tintin depth');
                    subplot(132); pcshow(reshape(pointCloud_source,width*height,3)); title('point cloud of source depth');
                    subplot(133); pcshow(reshape(pointCloud_pu,width*height,3)); title('point cloud of phase unwrapped depth');
                end


                if is_visualizing
                    pause(10);
                end    
                
                
            end


    end

%    end 
