clc;clear;close all

fidx = 230;
load(['../logs_outputs/0629_corr_imgs_model_ed2_corr_prelu/bathroom_all_' num2str(fidx) '_depth_pred.mat'])
corr_imgs = x;
freqVec = (10:30:160) * 1e6;
maxd = 10;
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
depths = PhaseImgsToDepths(freqVec, phase_imgs(1:4,:,:), delayVec/2);

figure; 
subplot(131); imagesc(depths); axis image; colorbar
load(['../data/bathroom_all_' num2str(fidx) '_depth_multiphase.mat']);
subplot(132); imagesc(depths); axis image; colorbar
load(['../data/bathroom_all_' num2str(fidx) '_depth_peak.mat']);
subplot(133); imagesc(peaks2depths); axis image; colorbar