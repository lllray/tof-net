% % given tintin depth and raw measurements
% % exact I, Q and depth from the binary
% % calculate depth_pu using phase unwrapping
% % prepare correct data format for network prediction (my ed and pix2pix)

clc; clear; close all

max_depth_norm = 10; % max depth for normalization to [0,1]
max_depth_vis = 6; % max depth for visualization
is_visualizing = true;
is_saving = true;
normalization = 2;
catamp = 1; % if concat amp image with imA
calib_phase_offset = 1; 
navg = 5; % number of measurements to average
output_folder = sprintf('phase_calibrated_norm2amp_rebuttal_mean%d', navg);

% date = {'0714'};
% freqs = [40,70];
% takes = 0:9;
% nc = 50;

% date = {'0715'};
% freqs = [40,70];
% takes = 0:7;
% nc = 50;

% date = {'0924'};
% freqs = [40,70];
% takes = 0:11;
% nc = 5;

% date = {'0926'};
% freqs = [40,70];
% takes = 0:42;
% nc = 20;

% date = {'0929'};
% freqs = [40,70];
% takes = 0:50;
% nc = 20;

% date = {'1018'};
% freqs = [40,70];
% takes = 0:1;
% nc = 20;

% date = {'1021'};
% freqs = [40,70];
% takes = 1:9;
% nc = 20;

date = {'1110'};
freqs = [40,70];
takes = 0:41;
nc = 20;

% date = {'0122'};
% freqs = [40,70];
% takes = 0:4;
% nc = 30;
% --
for idate = 1:numel(date)
    folder = ['~/bag/tintin_EE367/data_tintin/meas_' date{idate}];
    mkdir(['~/bag/tintin_EE367/data_tintin/meas_' date{idate} '/' output_folder]);
    for itakes = 1:numel(takes)
        fprintf('processing: %s, %d\n', date{idate}, itakes);
        %%%%%%%%% process depth
        fnd = sprintf('%s/%s_depth_%d',folder,date{idate},takes(itakes));
        if ~exist(fnd,'file')
            continue;
        end
        fileID = fopen(fnd);
        data = fread(fileID,Inf,'float');
        if size(data) ~= 153603*nc
            continue;
        end
        data = reshape(data,153603,nc);
        data_mean = mean(data(:,end-navg+1:end),2);
        depth = data_mean(4:3+240*320);
        depth = reshape(depth,320,240)';
%         depth = medfilt2(depth); % smooth it a bit
        if is_visualizing
            figure(1);
            suptitle(fnd);
            subplot(221); imagesc(depth); axis image; colorbar; title('depth');
            subplot(223); plot(squeeze(depth(end/2,:))); legend('mid row'); 
            subplot(224); plot(squeeze(depth(:,end/2))); legend('mid col'); 
        end
        fclose(fileID);
        if is_saving
            save(sprintf('%s/%s/%s_depth_%d.mat',folder,output_folder,date{idate},takes(itakes)),'depth');
            imwrite(depth/max_depth_vis, sprintf('%s/%s/%s_depth_%d.png',folder,output_folder,date{idate},takes(itakes)),'Bitdepth',16);
        end
        
        %%%%%%%%% process correlation meas
        is_corr_good = true;
        h = zeros(numel(freqs)*2,240,320);
        for ifreqs = 1:numel(freqs)
            fnc = sprintf('%s/%s_raw_%dMHz_%d',folder,date{idate},freqs(ifreqs),takes(itakes));
            if ~exist(fnc,'file')
                is_corr_good = false;
                continue;
            end
            fileID = fopen(fnc);
            data = fread(fileID,Inf,'float');
            if size(data) ~= 153603*nc
                is_corr_good = false;
                continue;
            end
            data = reshape(data,153603,nc);
            data_mean = mean(data(:,end-navg+1:end),2);
            phase_Q = data_mean(4:3+240*320);
            phase_I = data_mean(4+240*320:end);
            I_Mat = reshape(phase_I,320,240)';
            Q_Mat = reshape(phase_Q,320,240)';
            if calib_phase_offset
                load ~/bag/tintin_EE367/src/CalibratePhaseOffsetReal;
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
                suptitle(fnc);
                subplot(221); imagesc(I_Mat); axis image; colorbar; title('I');
                subplot(222); imagesc(Q_Mat); axis image; colorbar; title('Q');
                subplot(223); imagesc(Amp); axis image; colorbar; title('amp');
                subplot(224); imagesc(Phase); axis image; colorbar; title('phase');
            end
            if is_saving
                save(sprintf('%s/%s/%s_amp_%d_%d.mat',folder,output_folder,date{idate},takes(itakes),freqs(ifreqs)),'Amp');
                save(sprintf('%s/%s/%s_phase_%d_%d.mat',folder,output_folder,date{idate},takes(itakes),freqs(ifreqs)),'Phase');
            end
            h(ifreqs,:,:) = I_Mat;
            h(ifreqs+numel(freqs),:,:) = Q_Mat;
            fclose(fileID);    
        end
        if ~is_corr_good
            continue;
        end
        %%%%%%%%% save corr for network prediction (my implementation of ed)
        corr_imgs = h;
        if is_saving
            save(sprintf('%s/%s/real_%s_all_%d_corr_imgs',folder,output_folder,date{idate},takes(itakes)),'corr_imgs');
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
        phase = angle(corr_imgs(1:end/2,:,:) + 1i*corr_imgs(end/2+1:end,:,:));
        phase = phase([1,end],:,:);
        freqsm = freqs/10*10e6;
        lambda = 3e8./freqsm;
        phase(phase<0) = 2*pi + phase(phase<0);
        tic;
        depth_pu = PhaseImgs2Depths(freqsm, phase, 0:0.02:10);
        toc;
        if is_saving
            save(sprintf('%s/%s/%s_depth_pu_%d.mat',folder,output_folder,date{idate},takes(itakes)),'depth_pu');
            imwrite(depth_pu/max_depth_vis, sprintf('%s/%s/%s_depth_pu_%d.png',folder,output_folder,date{idate},takes(itakes)),'Bitdepth',16);
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
        if is_saving
            save(sprintf('%s/%s/%s_pix2pix_test_real_%d',folder,output_folder,date{idate},takes(itakes)),'im_pair');
        end
        
        %%%%%%%%% visualize point clouds
        pointCloud = depthToPointCloud(depth);
        pointCloud_pu = depthToPointCloud(depth_pu);
        if is_visualizing
            figure(numel(freqs)+2);
            subplot(121); pcshow(reshape(pointCloud,320*240,3)); title('point cloud of tintin depth');
            subplot(122); pcshow(reshape(pointCloud_pu,320*240,3)); title('point cloud of phase unwrapped depth');
        end
        if is_saving
            save(sprintf('%s/%s/%s_pointcloud_%d',folder,output_folder,date{idate},takes(itakes)),'pointCloud');
            save(sprintf('%s/%s/%s_pointcloud_pu_%d',folder,output_folder,date{idate},takes(itakes)),'pointCloud_pu');
        end
        
        if is_visualizing
            pause(10);
            %for ifig = 1:numel(freqs)+2, clf(ifig); end
        end
    end
end

%close all

