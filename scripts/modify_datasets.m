clc; clear; close all

save_modeify = 1;

datasets_dir = '/media/lixin/7A255A482B58BC84/lx/0429/deep_tof_datasets/train_data2/train'
data_name ='000980.mat'

if save_modeify
    out_path = sprintf('%s/../filter_train',datasets_dir)
    mkdir([out_path]);
end


datasets_list = dir(datasets_dir);
n = length(datasets_list);
% for i=3:n
%     data_name = datasets_list(i).name
%     load(sprintf('%s/%s',datasets_dir,data_name))
%     dist_gt = im_pair(:,:,end/2+1:end) ;
% 
%     dist_gt = 1./(dist_gt+1);
%     nan_index = find(isnan(dist_gt));
%     %dist_gt(nan_index) = 0;
%     dist_gt(dist_gt>=1) = 1;
%     im_pair(:,:,end/2+1:end) = dist_gt;
%     if save_modeify
%        save(sprintf('%s/%s',out_path,data_name),'im_pair');
%     end
% end
for i=3:n
    data_name = datasets_list(i).name
    load(sprintf('%s/%s',datasets_dir,data_name))
    dist_gt = im_pair(1,:,end/2+1:end) ;
    nan_index = find(isnan(dist_gt));
    if length(nan_index) >= 180*240*0.1
        length(nan_index)
        continue
    end
    if save_modeify
       save(sprintf('%s/%s',out_path,data_name),'im_pair');
    end
end