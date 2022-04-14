clc;clear;close all
rng(0);
for j = 1:10
    fns = '/Users/ssu/GitHub/pbrt-v3-scenes/pavilion/pavilion_tof.pbrt';
    fnt = sprintf('/Users/ssu/GitHub/pbrt-v3-scenes/pavilion/pavilion_tof_%d.pbrt',j);
    strs = '"string type" [ "matte" ]';
    strt = @(a) sprintf('"string type" [ "matte" ] "rgb Kd" [ %.2f %.2f %.2f ] ', a, a, a);
    % Read txt into cell A
    fid = fopen(fns,'r');
    i = 1;
    tline = fgetl(fid);
    A{i} = tline;
    while ischar(tline)
        i = i+1;
        tline = fgetl(fid);
        % Change cell A
        if ischar(tline) && contains(tline,strs)
            albedo = rand()/2+0.3;
            tline = strrep(tline,strs,strt(albedo));
        end
        A{i} = tline;
    end
    fclose(fid);
    % Write cell A into txt
    fid = fopen(fnt, 'w');
    for i = 1:numel(A)
        if A{i+1} == -1
            fprintf(fid,'%s', A{i});
            break
        else
            fprintf(fid,'%s\n', A{i});
        end
    end
end