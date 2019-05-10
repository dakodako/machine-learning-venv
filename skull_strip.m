clc;    % Clear the command window.
close all;  % Close all figures (except those of imtool.)
clear;  % Erase all existing variables. Or clearvars if you want.
% workspace;  % Make sure the workspace panel is showing.
format long g;
format compact;
fontSize = 24;

mp2 = niftiread('../MP2RAGE2/mp2_292.nii.gz');

for i = 1:51
    slice = mp2(:,:,128-25 + i);
    %[pixelCount, grayLevels] = imhist(slice);
    %bar(grayLevels, pixelCount);
    binaryImage = slice > 200;
    binaryImage = bwareaopen(binaryImage, 20);
    %figure
    %imagesc(binaryImage)
    %colormap gray
    %axis image
    binaryImage(end,:) = true;
    binaryImage = imfill(binaryImage, 'holes');
    %figure
    %imagesc(binaryImage)
    %colormap gray
    %axis image
    s = regionprops(binaryImage, 'Area', 'PixelList');
    [~,ind] = max([s.Area]);
    pix = sub2ind(size(binaryImage), s(ind).PixelList(:,2), s(ind).PixelList(:,1));
    out = zeros(size(binaryImage));
    out(pix) = slice(pix);
    %figure
    imagesc(out)
    axis image
    colormap gray
end

% Get the dimensions of the image.
% numberOfColorBands should be = 1.
[rows, columns] = size(grayImage);

% Display the original gray scale image.
subplot(2, 3, 1);
imshow(grayImage, []);
axis on;
title('Original Grayscale Image', 'FontSize', fontSize);
% Enlarge figure to full screen.
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
% Give a name to the title bar.
set(gcf, 'Name', 'Demo by ImageAnalyst', 'NumberTitle', 'Off')
% Let's compute and display the histogram.
[pixelCount, grayLevels] = imhist(grayImage);
subplot(2, 3, 2);
bar(grayLevels, pixelCount);
grid on;
title('Histogram of original image', 'FontSize', fontSize);
xlim([0 grayLevels(end)]); % Scale x axis manually.
% Crop image to get rid of light box surrounding the image
grayImage = grayImage(3:end-3, 4:end-4);
% Threshold to create a binary image
binaryImage = grayImage > 15;
% Get rid of small specks of noise
binaryImage = bwareaopen(binaryImage, 10);
% Display the original gray scale image.
subplot(2, 3, 3);
imshow(binaryImage, []);
axis on;
title('Binary Image', 'FontSize', fontSize);
% Seal off the bottom of the head - make the last row white.
binaryImage(end,:) = true;
% Fill the image
binaryImage = imfill(binaryImage, 'holes');
subplot(2, 3, 4);
imshow(binaryImage, []);
axis on;
title('Cleaned Binary Image', 'FontSize', fontSize);
% Erode away 15 layers of pixels.
se = strel('disk', 15, 0);
%binaryImage = imerode(binaryImage, se);
binaryImage = binaryImage;
subplot(2, 3, 5);
imshow(binaryImage, []);
axis on;
title('Eroded Binary Image', 'FontSize', fontSize);
% Mask the gray image
finalImage = grayImage; % Initialize.
finalImage(~binaryImage) = 0;
subplot(2, 3, 6);
imshow(finalImage, []);
axis on;
title('Skull stripped Image', 'FontSize', fontSize);
msgbox('Done with demo');