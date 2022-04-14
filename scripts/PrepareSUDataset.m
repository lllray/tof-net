addpath('/Users/ssu/Documents/MATLAB/Enhanced_rdir');

folder_syn = '/Users/ssu/GitHub/LearningMultiPath/local/data_trainingset_C_cos';
folder_real = '/Users/ssu/GitHub/LearningMultiPath/local/data_tintin';
folder_dataset = '/Users/ssu/GitHub/LearningMultiPath/src/GAN/datasets/mpi_correction_corrnorm2depth_s+u';

% % first process synthetic data
d = rdir([folder_syn, '/*_all*corr_imgs.mat']);
ntrain = 1;
ntest = 1;
for i = 1:numel(d)
    load(d(i).name);
    % normalize the amp, and select low/high freq idx
    corr_imgs_abs = abs(corr_imgs(1:end/2,:,:) + 1i*corr_imgs(end/2+1:end,:,:));
    corr = corr_imgs ./ cat(1, corr_imgs_abs, corr_imgs_abs);
    corr = (corr + 1) / 2; % convert from [-1,1] to [0,1]
    corr = corr([1,end/2,end/2+1,end],:,:); % choose some freq (if numel(freqs)>2)
    % save to folder
    im = corr;
    k = rand();
    if k > 0.1
        save(sprintf('%s/trainA/%d.mat', folder_dataset, ntrain), 'im');
        ntrain = ntrain + 1;
    else
        save(sprintf('%s/testA/%d.mat', folder_dataset, ntest), 'im');
        ntest = ntest + 1;
    end
end

% % then process real data
d = rdir([folder_real, '/**/real*corr_imgs.mat']);
ntrain = 1;
ntest = 1;
for i = 1:numel(d)
    load(d(i).name);
    % normalize the amp, and select low/high freq idx
    corr_imgs_abs = abs(corr_imgs(1:end/2,:,:) + 1i*corr_imgs(end/2+1:end,:,:));
    corr = corr_imgs ./ cat(1, corr_imgs_abs, corr_imgs_abs);
    corr = (corr + 1) / 2; % convert from [-1,1] to [0,1]
    corr = corr([1,end/2,end/2+1,end],:,:); % choose some freq (if numel(freqs)>2)
    % save to folder
    im = corr;
    k = rand();
    if k > 0.1
        save(sprintf('%s/trainB/%d.mat', folder_dataset, ntrain), 'im');
        ntrain = ntrain + 1;
    else
        save(sprintf('%s/testB/%d.mat', folder_dataset, ntest), 'im');
        ntest = ntest + 1;
    end
end