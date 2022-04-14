clc; clear; close all

expt = {'results4paper/SRA_results_all/'};
mseall = zeros(numel(expt), 250);
absall = zeros(numel(expt), 250);
ssimall = zeros(numel(expt), 250);
mses = zeros(numel(expt), 1);
abss = zeros(numel(expt), 1);
ssims = zeros(numel(expt), 1);
% pair = zeros(numel(expt),25,240,320*2);
load mpiratio;
impi = -1;
for i = 1:numel(expt)
    ipair = 1;
    fprintf('%d/%d ... ', i, numel(expt));
    folder1 = expt{i};
    folder2 = sprintf('../../src/GAN/results/%s/latest_test/images/real_B/','expt200');
    fn = dir([folder1 '*.mat']);
    sz = min(250, numel(fn));
    mse_ = zeros(sz,1);
    abs_ = zeros(sz,1);
    ssim_ = zeros(sz,1);
    for j = 1:sz
        % output
        load([folder1, fn(j).name]);
        x1 = squeeze(depthSRA);
        % GT
        load([folder2, fn(j).name]);
        x2 = squeeze(x(1,:,:))*10;
        % ignore nan/inf regions
        %x1(x2==0) = 0
        mask = (x2==0);
        se = strel('disk',5);
        maskd = imdilate(squeeze(mask),se);
        x1(maskd==1) = 0;
        x2(maskd==1) = 0;
    %	x1 = zeros(size(x1));
        % ignore boundary
        %x1 = x1(20:end-20,20:end-20);
        %x2 = x2(20:end-20,20:end-20);
        % visualize
        if 0
            figure(1); 
            subplot(121); imagesc(x1); colorbar
            subplot(122); imagesc(x2); colorbar
            suptitle(sprintf('%d, %d', i, j));
            pause(0.2);
        end
    % 	if mod(j,10)==0
    % 		pair(i,ipair,1:size(x1,1),[1:size(x1,2) 321:320+size(x1,2)]) = [x1 x2];
    % 		ipair = ipair+1;
    % 	end
        % metric
        if impi == -1 %overall
            mse_(j) = sqrt(sum((x1(:)-x2(:)).^2) / (size(x1,1)*size(x1,2)));
            abs_(j) = sum(abs(x1(:)-x2(:))) / (size(x1,1)*size(x1,2));
            ssim_(j) = ssim(single(x1),x2);
        else %0,1,2
            % and for each mpi strength (0,1,2)
            mpimask = (squeeze(mpi_strength(j,:,:))==impi);
            x1(mpimask==0) = 0;
            x2(mpimask==0) = 0;
            mse_(j) = sqrt(sum((x1(:)-x2(:)).^2) / numel(find(mpimask==1)) );
            abs_(j) = sum(abs(x1(:)-x2(:))) / numel(find(mpimask==1));
            ssim_(j) = ssim(single(x1),x2);
        end
    end
    mse_(isnan(mse_)) = 0;
    abs_(isnan(abs_)) = 0;
    ssim_(isnan(ssim_)) = 0;
    mseall(i,1:sz) = mse_;
    absall(i,1:sz) = abs_;
    ssimall(i,1:sz) = ssim_;
    mses(i) = sum(mse_)/sz;
    abss(i) = sum(abs_)/sz;
    ssims(i) = sum(ssim_)/sz;
    fprintf('SSIM: %04.4f, MSE: %04.4f, ABS: %04.4f\n', ssims(i), mses(i), abss(i));
end
% mses = mses(:,1:45);
% figure; plot(mseall', 'LineWidth', 1.5); legend(expt);
% [minval, minidx] = min(mses);
% figure; histogram(minidx);
% figure; stem(mses); xticks(1:numel(expt)); xticklabels(expt); view([90 -90]);
% figure; stem(abss); xticks(1:numel(expt)); xticklabels(expt); view([90 -90]);
