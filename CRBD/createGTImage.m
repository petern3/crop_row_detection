function [ GTImage ] = createGTImage(imageName, showSaveFlag, imagePath, GTDataPath)

%createGTImage Creates a crop row ground truth image
%   createGTImage function loads an image defined by imageName and creates
%   a crop row ground truth image according to its ground truth data which
%   must be named [imageName].crp
%
%
% Syntax:
%   GTImage = createGTImage(imageName, showSaveFlag, imagePath, GTDataPath);
%   GTImage = createGTImage(imageName, showSaveFlag, imagePath);
%   GTImage = createGTImage(imageName, showSaveFlag);
%   GTImage = createGTImage(imageName);
%   
%
% Inputs:
%   - imageName: the name of the image for which a crop row ground truth
%   image is created. Ground truth data for the image must be named 
%   [imageName].crp
%
%   - showSaveFlag (optional): flag for displaying and/or saving the 
%   created crop row ground truth image. If this flag is not set, function
%   returns the created image but it doesn't display the image and doesn't
%   save it. Valid values for the parameter are 1, 2 and 3. 
%   According to the set value, the function does the following:
%       - 1 : displays the created image
%       - 2 : saves the created image
%       - 3 : displays and saves the created image
%
%   - imagePath (optional): absolute path to the image folder if it is not
%   in the same folder as the function.
%
%   - GTDataPath (optional): absolute path to the ground truth data if it
%   is not in the same folder as function.
%
% Output:
%   - GTImage: Crop row ground truth image.
%
% Example:
%   GTImage = createGTImage('crop_row_037.jpg', 1, 'C:\Images', 'C:\GTData');
%   GTImage = createGTImage('crop_row_037.jpg', 1, 'C:\Images');
%   GTImage = createGTImage('crop_row_037.jpg', 1);
%   GTImage = createGTImage('crop_row_037.jpg');
%   createGTImage('crop_row_001.JPG', 2, 'Documents/CRBD/Images', 'Documents/CRBD/GT data');

%
% Copyright (c), Ivan Vidoviæ and Robert Cupec
% Faculty of Electrical Engineering Osijek
% J.J. Strossmayer University of Osijek
% Croatia
% ividovi2(at)etfos.hr; rcupec(at)etfos.hr
%
% Permission is hereby granted, free of charge, to any person obtaining
% a copy of this Software without restriction, subject to the following
% conditions:
% The above copyright notice and this permission notice should be included
% in all copies or substantial portions of the Software.
%
% The Software is provided "as is," without warranty of any kind.
%
% Created: June 19, 2015
% Last modified: June 30, 2015
%


%% Check input arguments
if nargin == 1
    showSaveFlag = 0;
    imagePath = imageName;
    GTDataPath = '';
elseif nargin == 2
    imagePath = imageName;
    GTDataPath = '';
elseif nargin == 3
    GTDataPath = '';
end

%% section

imageHeight = 240;
imageWidth = 320;

halfWidth = imageWidth/2;

%create image path
if(~strcmp(imagePath, imageName))
    imagePath = strcat(imagePath, '/', imageName);
end

%load original image
GTImage = imread(imagePath);

%create ground truth data file name
GTDataName = strcat(imageName(1 : strfind(imageName, '.')), 'crp');

%create ground truth data path
if(~strcmp(GTDataPath,''))
    GTDataPath = strcat(GTDataPath, '/', GTDataName);
else
    GTDataPath = GTDataName;
end

%load groun truth data
GT_CRP = load(GTDataPath);

%first image row where crop rows are present
v0 = imageHeight - size(GT_CRP, 1) + 1;

for v = v0 : imageHeight;
    
    %load c and d parameter for v-th image row
    c = GT_CRP(v - v0 + 1, 1);
    d = GT_CRP(v - v0 + 1, 2);
    
    %calculate start and stop index
    kStart = floor(-(halfWidth + c) / d) + 1;
    kEnd = floor((halfWidth - c) / d);
    
    for k = kStart : kEnd;
        
        %calculate pixel position
        u = int32(floor(c + k * d + halfWidth)) + 1; % + 1 because image indexes is not zero based
        
        %set GTimage pixel to red
        GTImage(v, u, 1) = 255;
        GTImage(v, u, 2) = 0;
        GTImage(v, u, 3) = 0;
    end;
end;

%create ground truth image name
GTImageName = strcat(imageName(1 : strfind(imageName, '.') - 1 ), '_GT.jpg');

%check showSaveFlag
switch(showSaveFlag)
    case 1
        imshow(GTImage);
    case 2
        imwrite(GTImage, GTImageName);
    case 3
        imshow(GTImage);
        imwrite(GTImage, GTImageName);
    end

end

