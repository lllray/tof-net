addpath('/Users/ssu/Documents/MATLAB/MatlabEXR');

datasets = {'bathroom','breakfast','contemporary-bathroom','pavilion','white-room'};
scales = [0.1, 0.2, 1, 0.5, 1];
folder1 = '/Users/ssu/GitHub/LearningMultiPath/local/depth_peak_0';
folder2 = '/Users/ssu/GitHub/LearningMultiPath/local/depth_zpass';
fn1 = @(i,j) sprintf('%s/%s_0_all_%d_depth_peak.mat',folder1,datasets{i},j);
fn2 = @(i,j) sprintf('%s/%s/Image%04d.exr',folder2,datasets{i},j);
diffall = [];
for i = 1:numel(datasets)
    for j = 1:250
        load(fn1(i,j)); 
        depth1 = peaks2depths; 
        depth2 = exrread(fn2(i,j)); 
        depth2 = mean(depth2,3)*scales(i);
        depth2(depth2>10) = 0;
        depth1(depth2==0) = 0; %ignore inf region
        diff = immse(depth1,depth2);
        diffall = [diffall; diff];
        figure(1);
        subplot(131); imagesc(depth1); axis image; 
        subplot(132); imagesc(depth2); axis image; 
        subplot(133); imagesc(depth1-depth2); axis image; caxis([-0.1,0.1]); title(diff);
        pause(0.1);
    end
end
        