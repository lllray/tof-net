clc;clear;close all

folder = '../data_tintin/meas_1110/phase_calibrated_norm1_depth2depth/';
fn = dir([folder '*.mat']);

for i = 1:numel(fn)
    load([folder fn(i).name]);
    im_pair = im_pair(1,:,:);
    im_pair(:,:,1:end/2) = im_pair(:,:,end/2+1:end);
    save([folder fn(i).name],'im_pair');
end