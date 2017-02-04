function [] = savePatchesfromContourPoints()

warning off;
% To save the patches around contour points for sending to CNN.

% Input:
% img_path - absolute path of the test image
% contour_points - (Kx2) matrix where each row corresponds to the
% co-ordinates of a contour point

% Written by Deepak Muralidharan, UCLA, 2016.
    
    img_path = 'test.jpg';
    I = imread(img_path);
    %I = imread('test.jpg');
    
    load('../tmp/ContourPoints.mat','ContourPoints');
    
    incorrect_rows = [];
    
    for i = 1:size(ContourPoints,1)
        center = ContourPoints(i,:);
        %disp(center)
        if ((center(2) > 32) && (center(2) < size(I,2) - 32) && (center(1) > 32) && (center(1) < size(I,1) - 32))
            subImage = I(center(1)-32: center(1)+31,center(2)-32:center(2)+31,:);
        %imshow(subImage,[]);
            file_name = sprintf('../tmp/patches/%d.mat',i);
            save(file_name,'subImage');
        else
            incorrect_rows = [incorrect_rows; i];
        end
    end
    
    ContourPoints(incorrect_rows,:) = [];
    
    save('../tmp/ContourPoints.mat','ContourPoints');
    
    
    quit
end