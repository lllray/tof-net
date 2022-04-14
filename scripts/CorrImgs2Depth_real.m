clc;clear;close all

load('real_all_2_corr_imgs.mat')
freqVec = (10:30:160) * 1e6;
maxd = 5;
nt = 250;
nf = numel(freqVec);
h = corr_imgs; 
h0mat = h(1:nf,:,:); %cos
h90mat = h(nf+1:end,:,:); %sin
corr_imgs = h0mat + 1i*h90mat;
phase_imgs = angle(corr_imgs);
for fi = 1:nf
    tmp = squeeze(phase_imgs(fi,:,:)<0);
    phase_imgs(fi,tmp) = 2*pi + phase_imgs(fi,tmp);
end
corr_imgs = cat(1,h0mat,h90mat);
delayVec = linspace(0,2*maxd,nt);
depths = PhaseImgs2Depths(freqVec, phase_imgs(1:4,:,:), delayVec/2);

figure;
for i = 1:12
    subplot(2,6,i); imagesc(squeeze(corr_imgs(i,:,:))); axis image; colorbar
end

figure; 
for i = 1:6
    subplot(1,6,i); imagesc(squeeze(phase_imgs(i,:,:))); axis image
end

figure; 
subplot(121); imagesc(depths); axis image
subplot(122); plot(depths(end/2,:));