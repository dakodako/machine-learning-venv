moving = p_slice;
fixed = squeeze(m_slice);
imshowpair(moving,fixed,'montage')
title('Unregistered')
[optimizer,metric] = imregconfig('multimodal');

movingRegisteredDefault = imregister(moving,fixed,'affine',optimizer,metric);
figure
imshowpair(movingRegisteredDefault,fixed)
title('A: Default Registration')

disp(optimizer)
disp(metric)

optimizer.InitialRadius = optimizer.InitialRadius/3.5;

movingRegisteredAdjustedInitialRadius = imregister(moving,fixed,'affine',optimizer,metric);
figure
imshowpair(movingRegisteredAdjustedInitialRadius,fixed)
title('B: Adjusted InitialRadius')

optimizer.MaximumIterations = 300;
movingRegisteredAdjustedInitialRadius300 = imregister(moving,fixed,'affine',optimizer,metric);
figure
imshowpair(movingRegisteredAdjustedInitialRadius300,fixed)
title('C: Adjusted InitialRadius, MaximumIterations = 300')

tformSimilarity = imregtform(moving,fixed,'similarity',optimizer,metric);
Rfixed = imref2d(size(fixed));

movingRegisteredRigid = imwarp(moving,tformSimilarity,'OutputView',Rfixed);
figure
imshowpair(movingRegisteredRigid, fixed)
title('D: Registration Based on Similarity Transformation Model')

movingRegisteredAffineWithIC = imregister(moving,fixed,'affine',optimizer,metric,...
    'InitialTransformation',tformSimilarity);
figure
imshowpair(movingRegisteredAffineWithIC,fixed)
title('E: Registration from Affine Model Based on Similarity Initial Condition')