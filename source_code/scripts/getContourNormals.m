function [] = getContourNormals()

warning off;

% To obtain the normal vector for each point on the contour

% Input:
% contour_points - (M x 2) where each row corresponds to a contour point

% Saves:
% Nk - (M x 2) matrix of contours

% Written by Deepak Muralidharan, UCLA, 2016.

load('../tmp/ContourPoints.mat','ContourPoints');
Nk = GetContourNormals2D(ContourPoints);
save('../tmp/Nk.mat','Nk');

quit 

end
