clc; clear; close all
my_data = importdata('/home/lixin/data/test_itof/_2022-04-29-14-38-46_pix2pix_test_real_1651214326513354.mat');
my_data2 = importdata('/home/lixin/data/test_itof/1651214326513354.mat');
my_data_value = my_data(:,60,80)
my_data2_value = my_data2(:,60,80)
my_data = permute(my_data(1,:,:),[2,3,1]);
my_data2 = permute(my_data2(1,:,:),[2,3,1]);
