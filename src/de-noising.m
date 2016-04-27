
% The attribute values of the image were calculated and stored in label, where X was the labeled image
label=regionprops(X, ‘all’);
% The regions with areas less than 240 were selected. The area threshold value was set as 240 which was obtained after many tests,
idx=find([label.Area]>240);
% The regions with areas less than 240 were removed
X1=ismember(X, idx);
% The relabeled image was presented as X2
label1=regionprops(X2,‘Centroid’);
