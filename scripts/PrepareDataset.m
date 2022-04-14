clc; clear; close all;
rng(0);
fn = {'bathroom','white-room','breakfast','contemporary-bathroom','pavilion'};
ntrain = 1;
% nval = 1;
ntest = 1;
for j = 1:numel(fn)
for k = 0:10
for i = 1:250
    fnc = sprintf('/home/shuochsu/Research/LearningMultiPath/data/%s_%d_all_%d_corr_imgs.mat', fn{j}, k, i);
    fnd = sprintf('/home/shuochsu/Research/LearningMultiPath/data/%s_%d_all_%d_depth_peak.mat', fn{j}, k, i);
    if ~exist(fnc,'file') || ~exist(fnd,'file')
        continue;
    end
    fprintf('Processing %s_%d_all_%d\n', fn{j}, k, i);
    load(fnc);
    load(fnd);
    % load corr images (40:10:70 MHz), normalize the amp, and select low/high freq idx
    corr_imgs_phase = angle(corr_imgs(1:end/2,:,:) + 1i*corr_imgs(end/2+1:end,:,:));
    corr_imgs_abs = abs(corr_imgs(1:end/2,:,:) + 1i*corr_imgs(end/2+1:end,:,:));
    corr = corr_imgs ./ cat(1, corr_imgs_abs, corr_imgs_abs);
    corr = (corr + 1) / 2; % convert from [-1,1] to [0,1]
    corr = corr([1,end/2,end/2+1,end],:,:);
    % load depth
    depth = reshape(peaks2depths, 1, size(peaks2depths,1), size(peaks2depths,2));
    depth = cat(1, depth, depth, depth, depth);
    depth = depth / 10; % convert to [0,1]
    % combine for pix2pix aligned data loader
    im_pair = cat(3,corr,depth);
    % split and save
    split = rand();
    if split > 0.1 %0.2
        save(sprintf('/home/shuochsu/Research/LearningMultiPath/src/GAN/datasets/mpi_correction_corrnorm2depth_albedoaug/train/%06d.mat',ntrain), 'im_pair');
        ntrain = ntrain + 1;
    % elseif split <=0.2 && split > 0.1
    %     save(sprintf('/Users/ssu/GitHub/LearningMultiPath/src/GAN/datasets/mpi_correction_corrnorm2depth/val/%04d.mat',nval), 'im_pair');
    %     nval = nval + 1;
    else
        save(sprintf('/home/shuochsu/Research/LearningMultiPath/src/GAN/datasets/mpi_correction_corrnorm2depth_albedoaug/test/%06d.mat',ntest), 'im_pair');
        ntest = ntest + 1;       
    end
end
end
end
