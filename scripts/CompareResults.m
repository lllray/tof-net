clc;clear;%close all
date = '0715';
freqs = 40:10:70;
take = 7;

load(sprintf('../data_tintin/real_%s_all_%d_corr_imgs.mat',date,take));
freqVec = freqs * 1e6;
maxd = 10;
nt = 250;
nf = numel(freqVec);
h0mat = corr_imgs(1:nf,:,:); %cos
h90mat = corr_imgs(nf+1:end,:,:); %sin
corr_imgs = h0mat + 1i*h90mat;
phase_imgs = angle(corr_imgs);
for fi = 1:nf
    tmp = squeeze(phase_imgs(fi,:,:)<0);
    phase_imgs(fi,tmp) = 2*pi + phase_imgs(fi,tmp);
end
corr_imgs = cat(1,h0mat,h90mat);
delayVec = linspace(0,2*maxd,nt);
depth_phaseunwarp = PhaseImgsToDepths(freqVec, phase_imgs, delayVec/2);

if 0
figure;
for i = 1:nf*2
    subplot(2,nf,i); imagesc(squeeze(corr_imgs(i,:,:))); axis image; colorbar
end

%figure(1); 
for i = 1:nf
    subplot(1,nf,i); imagesc(squeeze(phase_imgs(i,:,:))); axis image
end
end

% extract depth from tintin output
fileID = fopen(sprintf('../data_tintin/%s_depth_%d',date,take));
data = fread(fileID,Inf,'float');
data = reshape(data,153603,50);
data_mean = mean(data,2);
depth = data_mean(4:3+240*320);
amp = data_mean(4+240*320:end);
depth_internal = reshape(depth,320,240)';

% extract the learned depth
load(sprintf('../logs_outputs/0714_corr_imgs_model_ed2_tintin_phase/real_%s_all_%d_depth_pred',date,take));
depth_learned = x;

max_depth = max([depth_internal(:); depth_phaseunwarp(:); depth_learned(:)]);
min_depth = min([depth_internal(:); depth_phaseunwarp(:); depth_learned(:)]);
% show depth from tintin output
subplot(221); imagesc(depth_internal); axis image; caxis([min_depth,max_depth]); colorbar; title('depth (internal)');
% show depth from phase unwrapping
subplot(222); imagesc(depth_phaseunwarp); axis image; caxis([min_depth,max_depth]); colorbar; title('depth (reconstructed)');
% show the learned depth
subplot(223); imagesc(depth_learned); axis image; caxis([min_depth,max_depth]); colorbar; title('depth (learned)');

% compare scanlines
subplot(224); 
plot(depth_internal(end/2,:)); hold on; 
plot(depth_phaseunwarp(end/2,:)); hold on;
plot(depth_learned(end/2,:)); hold off
legend('depth (internal)', 'depth (reconstructed)', 'depth (learned)');
title('scanline (middle row)');

% save pdf
save2pdf(sprintf('%s_%d.pdf',date,take));