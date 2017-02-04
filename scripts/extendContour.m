function [] = extendContour()

warning off;
% To update the contour points using the equation
% C_{i+1} <- C_{i} + step*Vi.Ni

% Input:
% contour_old - (Kx2) old set of contour points (pixel co-ordinates).
% i = 1,...,K (where K is the total set of curve points
% Vk - (Kx2) gradient vectors for each contour point (1x2) predicted from CNN,
% i = 1,...,K
% Nk - (Kx2) normal vectors for each contour point (1x2), i = 1,...,K
% step size - ideally set to 0.5

% Written by Deepak Muralidharan, UCLA, 2016.

step_size = 0.5;

load('../tmp/ContourPoints.mat','ContourPoints');
load('../tmp/Vk.mat','Vk');
load('../tmp/Nk.mat','Nk');

Vk_interchanged = [Vk(:,2), Vk(:,1)];

tmp = step_size*(dot(Vk_interchanged,Nk,2).*Nk);
tmp(:,1) = -tmp(:,1);

ContourPoints = ContourPoints + tmp;
ContourPoints = double(ContourPoints);
save('../tmp/ContourPoints.mat','ContourPoints');

quit

end