clc; clear; close all
datasets_folder = '/home/lixin/data/test_itof';
output_folder = sprintf('impair');
width = 240;
height = 180;
fnd = sprintf('%s/output_0920.txt',datasets_folder);
my_data = importdata(fnd).';
[d_r,d_c] = size(my_data)
for i = 1:d_c
    data = my_data(:,i);
    timestamp = data(1);
    data = data(4:3+height*width*5);
    data = reshape(data,5,width,height);
    data = permute(data,[1,3,2]);
    im_pair = cat(3,data,data);
    im_pair_value = im_pair(:,60,80)
    im_pair_debug = permute(im_pair(1,:,:),[2,3,1]);
    save(sprintf('%s/ks_land_test_0920/test/%ld',datasets_folder,timestamp),'im_pair');
end

%验证
% clc; clear; close all
% my_data = importdata('/home/lixin/data/test_itof/_2022-04-29-14-38-46_pix2pix_test_real_1651214326513354.mat');
% my_data2 = importdata('/home/lixin/data/test_itof/1651214326513354.mat');
% my_data_value = my_data(:,60,80)
% my_data2_value = my_data2(:,60,80)
% my_data = permute(my_data(1,:,:),[2,3,1]);
% my_data2 = permute(my_data2(1,:,:),[2,3,1]);
