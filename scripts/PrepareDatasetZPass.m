clc; clear; close all;

% blender z pass
singlefreq = 0;
normalization = 0; % 0: no amp normalization; 1: apply image-wise normalization for each freq; 2: apply pixel-wise normalization
phase = 1; % 0: use input corr img as is; 1: convert to phase map
catamp = 0; % if concat amp image with imA
addpath('/home/shuochsu/Research/LearningMultiPath/src/MatlabEXR');
rng(0);
fn = {'bathroom','breakfast','contemporary-bathroom','pavilion','white-room'};
scale = [0.1, 0.2, 1, 0.5, 1];
ntrain = 1;
% nval = 1;
ntest = 1;
for ifn = 1:numel(fn)
for ial = 0:10
for iview = 1:250
    fnc = sprintf('/home/shuochsu/Research/LearningMultiPath/data/%s_%d_all_%d_corr_imgs.mat', fn{ifn}, ial, iview);
    fnd = sprintf('/home/shuochsu/Research/LearningMultiPath/data/depth_zpass/%s/Image%04d.exr', fn{ifn}, iview);
    if ~exist(fnc,'file') || ~exist(fnd,'file')
        continue;
    end
    fprintf('Processing %s_%d_all_%d\n', fn{ifn}, ial, iview);

    % load corr images (40:10:70 MHz), normalize the amp, and select low/high freq idx
    load(fnc);
    nf = size(corr_imgs,1)/2;
    corr_imgs_phase = angle(corr_imgs(1:nf,:,:) + 1i*corr_imgs(nf+1:end,:,:));
    corr_imgs_abs = abs(corr_imgs(1:nf,:,:) + 1i*corr_imgs(nf+1:end,:,:));
    
    % normalization (should not change the contrast by substracting minamp)
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
    
    % choose freq pair, and use corr or phase as imA
    if ~phase
	if ~singlefreq
        	imA = corr([1,nf,nf+1,end],:,:);
	else
		imA = corr([1,nf+1],:,:);
	end
    else
        imA = corr_imgs_phase([1,nf],:,:);
        imA = (imA/pi+1)/2; % convert from [-pi,pi] to [0,1]
    end
    if catamp
        tmp = corr_imgs_abs(1,:,:);
        minamp = min(tmp(:));
        maxamp = max(tmp(:));
        amp = tmp-minamp;
        if maxamp > minamp
            amp = amp / (maxamp-minamp);
        end
        imA = cat(1,imA,amp);
    end
    
    % load depth
    depth = exrread(fnd);
    depth = mean(depth,3)*scale(ifn);
    depth(depth>10) = Inf;
    depth = reshape(depth, 1, size(depth,1), size(depth,2));
    if ~phase
	if ~singlefreq
        	depth_ = cat(1, depth, depth, depth, depth);
	else
		depth_ = cat(1, depth, depth);
	end
    else
        depth_ = cat(1, depth, depth);
    end
    if catamp
        depth_ = cat(1, depth_, depth);
    end
    depth_ = depth_ / 10; % convert to [0,1]

    % combine for pix2pix aligned data loader
    im_pair = cat(3,imA,depth_);

    % split and save
    split = rand();
    if split > 0.1 %0.2
        save(sprintf('/home/shuochsu/Research/LearningMultiPath/src/GAN/datasets/mpi_correction_phase2depth_albedoaug_zpass/train/%06d.mat',ntrain), 'im_pair');
        ntrain = ntrain + 1;
%     elseif split <=0.2 && split > 0.1
%         save(sprintf('/home/shuochsu/Research/LearningMultiPath/src/GAN/datasets/mpi_correction_corrnorm2depth/val/%04d.mat',nval), 'im_pair');
%         nval = nval + 1;
    else
        save(sprintf('/home/shuochsu/Research/LearningMultiPath/src/GAN/datasets/mpi_correction_phase2depth_albedoaug_zpass/test/%06d.mat',ntest), 'im_pair');
        ntest = ntest + 1;       
    end
end
end
end
