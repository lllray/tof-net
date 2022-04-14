clc;clear

rx = @(a) [1 0 0 0; 0 cos(a) -sin(a) 0; 0 sin(a) cos(a) 0; 0 0 0 1];
ry = @(a) [cos(a) 0 sin(a) 0; 0 1 0 0; -sin(a) 0 cos(a) 0; 0 0 0 1];
rz = @(a) [cos(a) -sin(a) 0 0; sin(a) cos(a) 0 0; 0 0 1 0; 0 0 0 1];
s = @(sc) [sc(1) 0 0 0; 0 sc(2) 0 0; 0 0 sc(3) 0; 0 0 0 1];

blender_loc = [3.28149 13.812 15.21961];
blender_rot = [70.647 0.449 59.618];
blneder_s = [1 1 1];
m = rz(blender_rot(3)/180*pi) * ry(blender_rot(2)/180*pi) * rx(blender_rot(1)/180*pi) * s(blneder_s);
m(1:3,4) = blender_loc;

m = [-1 0 0 0; 0 0 1 0; 0 -1 0 0; 0 0 0 1] * m; % convert to pbrt coord

% reference: https://github.com/mmp/pbrt-v2/blob/master/exporters/blender/pbrtBlend.py
pos = squeeze(m(:,4));
forwards = -squeeze(m(:,3));
up = squeeze(m(:,2));
target = pos+forwards;

fprintf('LookAt %f %f %f %f %f %f %f %f %f\n',...
    pos(1),pos(2),pos(3),...
    target(1),target(2),target(3),...
    up(1),up(2),up(3));