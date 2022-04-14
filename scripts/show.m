clc;clear;%close all
load('~/ROS/DeepEnd2End/DeepToF_release_0.1/src/GAN/results/my_resnet_lite_my_imageGAN_128_pretrained/latest_test/images/fake_B/000001.mat') %results are stored as .mat files under [results_folder]/fake_B
x = x*10 %scale from [0,1] back to distance in meters
imagesc(squeeze(x(1,:,:))) %display tofnet output