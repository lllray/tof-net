% % given tintin depth and raw measurements
% % exact I, Q and depth from the binary
% % calculate depth_pu using phase unwrapping
% % prepare correct data format for network prediction (my ed and pix2pix)

clc; clear; close all

max_depth_norm = 10; % max depth for normalization to [0,1]
max_depth_vis = 10; % max depth for visualization
is_visualizing = false;
is_saving_debug = false;
is_saving_depth_pu = false;
is_saving_release = true;
normalization = 2;
catamp = 1; % if concat amp image with imA
calib_phase_offset = 0; 
navg = 1; % number of measurements to average
datasets_folder = '/home/lixin/data/itof_test/230416-02'; % 文件夹中包含 itof_output_37650000.txt itof_output_45180000.txt itof_output_depth.txt
depth_folder = sprintf('phase_calibrated_norm2amp_rebuttal_mean%d', navg);
data_folder = sprintf('impair_mean%d', navg);

medfilt_size = 1;
use_my = 1;
use_depth_as_gt = 1;

    %-----  my data  ---------
    date = {''};
    freqs = [45180000,37650000];
    takes = 0;
    width = 240;
    height = 180;
    % mydata_a = importdata('~/bag/tintin_EE367/my_data/itof_output_a.txt').';
    % mydata_b = importdata('~/bag/tintin_EE367/my_data/itof_output_b.txt').';

    for idate = 1:numel(date)
        folder = [datasets_folder date{idate}];
        mkdir([folder '/' depth_folder]);
        mkdir([folder '/' data_folder]);
        itakes = 1;
            
            %%%%%%%%% process depth
            fnd = sprintf('%s/itof_output_depth.txt',folder);
            if ~exist(fnd,'file')
                printf("not find depth data")
               return;
            end
            depth_data = importdata(fnd).';
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
            [raw_r,raw_c,raw_n] = size(raw_data)
            if raw_c ~= d_c
                printf("depth data num is not eual raw data num")
            end
            for i = 1:d_c
                depth = depth_data(:,i);
                depth = depth(4:3+height*width);
                depth = reshape(depth,width,height)';
                depth(depth<=0.0) = nan; % is importan to set nan
                if medfilt_size > 1
                depth = medfilt2(depth,[medfilt_size,medfilt_size]);
                end
                if is_visualizing
                figure(1);
                suptitle(fnd);
                subplot(221); imagesc(depth); axis image; colorbar; title('depth');
                subplot(223); plot(squeeze(depth(end/2,:))); legend('mid row'); 
                subplot(224); plot(squeeze(depth(:,end/2))); legend('mid col'); 
                end
                if is_saving_release
                    save(sprintf('%s/%s/%s_depth_%d.mat',folder,depth_folder,date{idate},i),'depth');
                    imwrite(depth/max_depth_vis, sprintf('%s/%s/%s_depth_%d.png',folder,depth_folder,date{idate},i),'Bitdepth',16);
                end
                
                h = zeros(numel(freqs)*2,height,width);
                for ifreqs = 1:numel(freqs)
                    raw = raw_data(:,i,ifreqs);
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
                        I_Mat = medfilt2(I_Mat,[medfilt_size,medfilt_size]);
                        Q_Mat = medfilt2(Q_Mat,[medfilt_size,medfilt_size]);
                        Amp = abs(I_Mat+1i*Q_Mat);
                        Phase = angle(I_Mat+1i*Q_Mat);
                        if is_visualizing
                            figure(ifreqs+1); 
                            suptitle(fnc);
                            subplot(221); imagesc(I_Mat); axis image; colorbar; title('I');
                            subplot(222); imagesc(Q_Mat); axis image; colorbar; title('Q');
                            subplot(223); imagesc(Amp); axis image; colorbar; title('amp');
                            subplot(224); imagesc(Phase); axis image; colorbar; title('phase');
                        end
                        if is_saving_debug
                            save(sprintf('%s/%s/%s_amp_%d_%d.mat',folder,depth_folder,date{idate},i,freqs(ifreqs)),'Amp');
                            save(sprintf('%s/%s/%s_phase_%d_%d.mat',folder,depth_folder,date{idate},i,freqs(ifreqs)),'Phase');
                        end
                        h(ifreqs,:,:) = I_Mat;
                        h(ifreqs+numel(freqs),:,:) = Q_Mat;
                end

                %%%%%%%%% save corr for network prediction (my implementation of ed)
                corr_imgs = h;
                if is_saving_debug
                    save(sprintf('%s/%s/real_%s_all_%d_corr_imgs',folder,depth_folder,date{idate},i),'corr_imgs');
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
                    tmp = corr_imgs_abs(1,:,:);
                    minamp = min(tmp(:));
                    maxamp = max(tmp(:));
                    amp = tmp-minamp;
                    if maxamp > minamp
                        amp = amp / (maxamp-minamp);
                    end
                    corr = cat(1,corr,amp);
                end
        %         for i = 1:size(corr,1)
        %             corr(i,:,:) = medfilt2(squeeze(corr(i,:,:))); % smooth it a bit
        %         end
                % calculate depth
                if use_depth_as_gt
                    depth_pu = depth;
                else    
                    phase = angle(corr_imgs(1:end/2,:,:) + 1i*corr_imgs(end/2+1:end,:,:));
                    phase = phase([1,end],:,:);
                    freqsm = freqs;
                    lambda = 3e8./freqsm;
                    %neg = phase <=0;
                    %phase =  (phase.*~neg + neg.*(-1.*phase + pi));
                    phase(phase<0) = 2*pi + phase(phase<0);
                    tic;
                    depth_phaseunwarp = PhaseImgs2Depths(freqsm, phase, 0:0.02:10);
                    depth_pu = depth_phaseunwarp;
                end

                if is_saving_depth_pu
                    phase = angle(corr_imgs(1:end/2,:,:) + 1i*corr_imgs(end/2+1:end,:,:));
                    phase = phase([1,end],:,:);
                    freqsm = freqs;
                    lambda = 3e8./freqsm;
                    %neg = phase <=0;
                    %phase =  (phase.*~neg + neg.*(-1.*phase + pi));
                    phase(phase<0) = 2*pi + phase(phase<0);
                    tic;
                    depth_phaseunwarp = PhaseImgs2Depths(freqsm, phase, 0:0.02:10);
                    save(sprintf('%s/%s/%s_depth_pu_%d.mat',folder,depth_folder,date{idate},i),'depth_phaseunwarp');
                    imwrite(depth_phaseunwarp/max_depth_vis, sprintf('%s/%s/%s_depth_pu_%d.png',folder,depth_folder,date{idate},i),'Bitdepth',16);
                end
                % normalize, assemble
                depth_pu_pair = reshape(depth_pu, 1, size(depth_pu,1), size(depth_pu,2));
                if ~catamp
                    depth_pu_pair = cat(1, depth_pu_pair, depth_pu_pair, depth_pu_pair, depth_pu_pair);
                else
                    depth_pu_pair = cat(1, depth_pu_pair, depth_pu_pair, depth_pu_pair, depth_pu_pair, depth_pu_pair);
                end
                depth_pu_pair = depth_pu_pair / max_depth_norm; % convert to [0,1]
                % combine for pix2pix aligned data loader
                im_pair = cat(3,corr,depth_pu_pair);
                % split and save
                if is_saving_release
                    save(sprintf('%s/%s/%s_pix2pix_test_real_%d',folder,data_folder,date{idate},i),'im_pair');
                end
                
                
                %%%%%%%%% visualize point clouds
                %need sensor param
                pointCloud = my_depthToPointCloud(depth,'no_calib');
                pointCloud_pu = my_depthToPointCloud(depth_pu,'no_calib');
                if is_visualizing
                    figure(numel(freqs)+2);
                    subplot(121); pcshow(reshape(pointCloud,width*height,3)); title('point cloud of tintin depth');
                    subplot(122); pcshow(reshape(pointCloud_pu,width*height,3)); title('point cloud of phase unwrapped depth');
                end
                if is_saving_debug
                    save(sprintf('%s/%s/%s_pointcloud_%d',folder,depth_folder,date{idate},i),'pointCloud');
                    save(sprintf('%s/%s/%s_pointcloud_pu_%d',folder,depth_folder,date{idate},i),'pointCloud_pu');
                end

                if is_visualizing
                    pause(10);
                end    
                
                
            end


    end

%    end 
