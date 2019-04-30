%{
p = niftiread('PETRA2/petra_292.nii.gz');
m = niftiread('MP2RAGE2/mp2_292.nii.gz');

%imgs_m = zeros(256,256,256);
%imgs_p = zeros(256,256,256);


%imgs_m(:,31:223,:) = m;
%imgs_p(:,31:223,:) = p;

combine = zeros(256,512,256);
combine(:,31:223,:) = m;
combine(:,287:479,:) = p;
%}
%niftiwrite(combine, 'dataset/p2m/combined_292.nii.gz')
img_selected = combine(:,:,128-80:128+80);
for i = 1:101
    img = img_selected(:,:,i);
end