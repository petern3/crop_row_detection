[a, map] = imread('pout.tif'); %MATLAB already has pout.tif, or use another tif file
b = imnoise(a, 'salt & pepper'); %add salt & pepper noise

c = medfilt2(b); % insert one line of code to median filter out the added salt & pepper noise

subplot(1,3,1), subimage(a)
subplot(1,3,2), subimage(b)
subplot(1,3,3), subimage(c)

%imshow(a,map);