% assemble [depth_pu, depth_z] pair, also to evaluate mpi strength

clc; clear; close all;

addpath('/home/shuochsu/Research/LearningMultiPath/src/scripts/MatlabEXR');
rng(0);
fn = {'bathroom','breakfast','contemporary-bathroom','pavilion','white-room'};
scale = [0.1, 0.2, 1, 0.5, 1];
ntrain = 1;
ntest = 1;
for ifn = 1:numel(fn)
for ial = 0:10
for iview = 1:250
    fnc = sprintf('/home/shuochsu/Research/LearningMultiPath/data/%s_%d_all_%d_depth_multiphase_2freqs.mat', fn{ifn}, ial, iview);
    fnd = sprintf('/home/shuochsu/Research/LearningMultiPath/data/depth_zpass/%s/Image%04d.exr', fn{ifn}, iview);
    if ~exist(fnc,'file') || ~exist(fnd,'file')
        continue;
    end
    fprintf('Processing %s_%d_all_%d\n', fn{ifn}, ial, iview);

    % load corr images (40:10:70 MHz), normalize the amp, and select low/high freq idx
    load(fnc);
    depth_pu = depths / 10; % convert to [0,1]
    
    % load gt depth
    depth = exrread(fnd);
    depth = mean(depth,3)*scale(ifn);
    depth(depth>10) = Inf;
    depth = depth / 10; % convert to [0,1]

    % combine for pix2pix aligned data loader
    depth_pu = reshape(depth_pu, 1, size(depth_pu,1), size(depth_pu,2));
    depth = reshape(depth, 1, size(depth,1), size(depth,2));
    im_pair = cat(3,depth_pu,depth);

    % split and save
    split = rand();
    if split > 0.1 %0.2
        save(sprintf('/home/shuochsu/Research/LearningMultiPath/src/GAN/datasets/mpi_correction_depth2depth_albedoaug_zpass/train/%06d.mat',ntrain), 'im_pair');
        ntrain = ntrain + 1;
    else
        save(sprintf('/home/shuochsu/Research/LearningMultiPath/src/GAN/datasets/mpi_correction_depth2depth_albedoaug_zpass/test/%06d.mat',ntest), 'im_pair');
        ntest = ntest + 1;       
    end
end
end
end
