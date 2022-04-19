% depth from simulated transient images by peak fitting
% update (10/1/17) now augmented with 10 albedos on 25 views and 3 more scenes 
clc; clear; 
addpath('MatlabEXR');
if strcmp(computer, 'MACI64')
    path = '/Users/ssu/Documents/Research/LearningMultiPath/data';
    path_out = '/Users/ssu/Documents/Research/LearningMultiPath/data';
else
%     path = '/home/shuochsu/Research/pbrt-v3-tof/Results';
%     path_out = '/home/shuochsu/Research/LearningMultiPath/data';
    path = '/home/lixin/ROS/DeepEnd2End/DeepToF_release_0.1/src/pbrt-v3-tof/Results';
    path_out = '/home/lixin/ROS/DeepEnd2End/DeepToF_release_0.1/data';
end

datasets = {'bathroom','white-room','breakfast','contemporary-bathroom','pavilion'}; 
dataType = {'all'}; % {'all', 'direct'};
albedos = {'','1','2','3','4','5','6','7','8','9','10'};
numCameraViews = 250;

is_extracting_data = 0;
is_visualizing_transient = 0;
is_peak_fitting = 0;
is_simulating_phase_imgs = 1;
is_simulating_depth_from_phase_imgs = 1;

is_visualize = 0;

for dti = 1:numel(dataType)
for di = 1:numel(datasets)
for ai = 1:numel(albedos)
for vi = 1:numCameraViews

fprintf('Processing %s (%s) at camera view %d of albedo %s\n', datasets{di}, dataType{dti}, vi, albedos{ai});

%%%% Main section of code (inside the interations) %%%%
if is_visualize
    close all;
end

if ai == 1
    folder = [path '/' datasets{di} '_' dataType{dti} '_' num2str(vi)];
    folder_out = [path_out '/' datasets{di} '_0_' dataType{dti} '_' num2str(vi)];
else
    folder = [path '/' datasets{di} '_' albedos{ai} '_' dataType{dti} '_' num2str(vi)];
    folder_out = [path_out '/' datasets{di} '_' albedos{ai} '_' dataType{dti} '_' num2str(vi)];
end

disp(folder);
if numel(dir([folder '/0*.exr'])) ~= 256
    fprintf('folder does not exist, skipped');
    continue;
end

if is_extracting_data
    nh = 240; nw = 320; nt = 256;
    hdrs = zeros(nh,nw,nt);
    for ti = 1:nt
%        display(ti);
        if strcmp(datasets{di},'white-room')
            hdr = exrread([folder '/' sprintf('%04d_%s.exr',ti-1,'whiteroom')]);
        else
            hdr = exrread([folder '/' sprintf('%04d_%s.exr',ti-1,datasets{di})]);
        end
%         rgb = tonemap(hdr);
        hdrs(:,:,ti) = mean(hdr,3);
    end
    save([folder_out '_transient.mat'], 'hdrs');
else
    load([folder_out '_transient.mat']);
end

if is_visualize && is_visualizing_transient
    % load([folder_out '_transient.mat']);
    maxIntensity = max(hdrs(:));
    minIntensity = min(hdrs(:));
    figure;
    for ti = 1:size(hdrs,3)
        imagesc(squeeze(hdrs(:,:,ti)),[minIntensity, maxIntensity]);
        colorbar;
        pause(0.01);
    end
end

% load([folder_out '_transient.mat']);
[nh,nw,nt] = size(hdrs);
scale = 1;
maxd = 10*scale;
if is_peak_fitting
    peaks = zeros(nh,nw);
    for ih = 1:nh
        for iw = 1:nw
            transient_pixel = squeeze(hdrs(ih,iw,:));
            p = PeakFitting(transient_pixel);
            if numel(p) == 0
                [~,p] = max(transient_pixel);
            end
            peaks(ih,iw) = p;
        end
    end
    peaks2depths = peaks/nt*maxd;
    save([folder_out '_depth_peak.mat'],'peaks2depths');
    if is_visualize
        figure; 
        subplot(121); imagesc(peaks2depths,[0,6]); axis image; colorbar;
        subplot(122); plot(peaks2depths(floor(nh/2),:)); ylim([0 6]);
        suptitle('Depth from peak fitting using transient images');
    end
end

% freqVec = (10:10:160) * 1e6;
freqVec = (40:10:70) * 1e6;
delayVec = linspace(0,2*maxd,nt); %20m round trip, scaled by some factor
if is_simulating_phase_imgs
    [phase_imgs, corr_imgs] = GetPhaseImgs(hdrs,freqVec,delayVec);
    % save([folder_out '_phase_imgs.mat'],'phase_imgs');
%     % select only a few corr_imgs from (10:30:160)
%     corr_imgs = corr_imgs([1,2,7,8,13,14,19,20,25,26,31,32],:,:);
%     save([folder_out '_corr_imgs.mat'],'corr_imgs','phase_imgs');
    if is_visualize
        figure; 
        for i = 1:numel(freqVec)
            subplot(4,5,i);
            imagesc(squeeze(phase_imgs(i,:,:))); 
            colorbar;
        end
    end
end

if is_simulating_depth_from_phase_imgs
    % load([folder_out '_phase_imgs.mat']);
    % use 2 freqs
    depths = PhaseImgs2Depths(freqVec([1,4]), phase_imgs([1,4],:,:), delayVec/2);
    save([folder_out '_depth_multiphase_2freqs.mat'],'depths');
    % use all 4 freqs
    depths = PhaseImgs2Depths(freqVec, phase_imgs, delayVec/2);
    save([folder_out '_depth_multiphase_4freqs.mat'],'depths');
    if is_visualize
        figure; 
        subplot(121); imagesc(depths,[0,6]); axis image; colorbar;
        subplot(122); plot(depths(floor(nh/2),:)); ylim([0 6]);
        suptitle('Depth from multi-freq phase unwrapping');
    end
end

%%%% End of main section of code (inside the interations) %%%%

end
end
end
end
