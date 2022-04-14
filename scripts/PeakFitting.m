% follow the direct/global separation algorithm from Di Wu et al.
% modified a bit
function p = PeakFitting(transient_pixel)
%% original implementation, good from streak images
%     dtp = conv(transient_pixel, [-1,1], 'same');
%     dtp(1) = 0;
%     dtp(end) = 0;
%     alpha = 0.05*max(dtp);
%     t_start = min(find(abs(dtp)>alpha));
%     beta = 0.0002;
%     t_middle = min(find(abs(dtp(t_start+1:end))<beta)) + t_start;
%     t_end = t_start + 2*(t_middle-t_start);
%     p = t_end;

%% modified
    dtp = conv(transient_pixel, [-1,1], 'same');
    dtp(1) = 0;
    dtp(end) = 0;
    alpha = 0.05*max(dtp);
    t_start = min(find(abs(dtp)>alpha));
    t_end = min(find(abs(dtp(t_start+1:end))>alpha)) + t_start;
    t_middle = floor((t_end+t_start)/2);
    p = t_middle;
end
    