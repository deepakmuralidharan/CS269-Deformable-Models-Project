function [] = extractTrainingData()

% Function to extract the training data and store the patches

% Written by Deepak Muralidharan, UCLA, 2016.
    

    clc;
    close all;
    clear all;
    addpath('/Users/deepakmuralidharan/Downloads/AOSLevelsetSegmentationToolboxM/AOSLevelsetSegmentationToolboxM'); %adding path for ac_reinit

    annotation_dir = '/Users/deepakmuralidharan/Downloads/DAVIS/Annotations/480p/goat-single';
    img_dir = '/Users/deepakmuralidharan/Downloads/DAVIS/JPEGImages/480p/goat-single';
    myFiles = dir(fullfile(annotation_dir,'*.png'));
    count = 0;
    for k = 1:length(myFiles)
        display(k);
        
        baseFileName = myFiles(k).name;
        disp(baseFileName);
        fullFileName = fullfile(annotation_dir, baseFileName);
        I = imread(fullFileName);
        [pathstr,name,~] = fileparts(baseFileName);
        
        % reading original image
        original_img_path = strcat(img_dir,'/',name,'.jpg');
        I_ori = imread(original_img_path);
        I(I > 0) = 255;
        % generating the SDM
        I = (I > 0) - 0.5; 
        I_sdf = ac_reinit(I);
        [Gx, Gy] = imgradientxy(I_sdf,'prewitt');
        
        %finding level lines
        level_lines = unique(I_sdf);
        level_lines = level_lines(abs(level_lines) < 20);
        display(length(level_lines));
        
        %sampling points in each level line and extracting the patch
        for i = 1:length(level_lines)
            %display(i);
            %disp(level_lines(i));
            [x,y] = find(abs(I_sdf-level_lines(i)) < 1e-4);
            r = randi([1 length(x)],round(length(x)),1);
            x_sampled = x(r);
            y_sampled = y(r);

            %generate 64x64 patches around the sampled points and find gradient
            %of SDM at each point
            for j = 1:length(x_sampled)
                %display(j);
                if ((x_sampled(j) > 32) && (x_sampled(j) < size(I_ori,1) - 32) && (y_sampled(j) > 32) && (y_sampled(j) < size(I_ori,2) - 32))
                    count = count + 1;
                    subImage = I_ori(x_sampled(j)-32: x_sampled(j)+31,y_sampled(j)-32:y_sampled(j)+31,:);
                    %IImage = I_sdf(x_sampled(j)-32: x_sampled(j)+32,y_sampled(j)-32:y_sampled(j)+32);
                    v = [Gx(x_sampled(j),y_sampled(j)); Gy(x_sampled(j),y_sampled(j))];
                    scale = I_sdf(x_sampled(j), y_sampled(j));
                    v = abs(scale)*v/norm(v);
                    v(2) = -v(2); %adjusting the direction
                    %display(v);
                    %imshow(IImage);
                    %pause;
                    img_name = sprintf('../../data/patch_%d.jpg',count);
                    file_name = sprintf('../../data/patch_%d.mat',count);
                    save(file_name,'subImage','v');
                    %imwrite(subImage,img_name);

                    %disp(v);
                    %figure;
                end
            end 
        end
        display(fullFileName);
    end
    
    quit
    
end