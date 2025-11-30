function [dsTrain, dsVal, dsTest] = load_data(dataPath, noiseFactor, imgSize)
    % 1. Create Datastore
    imds = imageDatastore(dataPath, 'IncludeSubfolders', true, 'LabelSource', 'none');
    
    % 2. Split Data (80-10-10)
    numFiles = numel(imds.Files);
    shuffledIndices = randperm(numFiles);
    nTrain = round(0.8 * numFiles);
    nVal = round(0.1 * numFiles);
    
    idxTrain = shuffledIndices(1:nTrain);
    idxVal = shuffledIndices(nTrain+1 : nTrain+nVal);
    idxTest = shuffledIndices(nTrain+nVal+1 : end);
    
    dsTrainRaw = subset(imds, idxTrain);
    dsValRaw = subset(imds, idxVal);
    dsTestRaw = subset(imds, idxTest);

    % 3. Define Noise Transformation
    augmenter = @(x) add_noise_transform(x, noiseFactor, imgSize);

    % 4. Create Combined Datastores (Input=Noisy, Response=Clean)
    dsTrain = transform(dsTrainRaw, augmenter);
    dsVal = transform(dsValRaw, augmenter);
    dsTest = transform(dsTestRaw, augmenter);
end

function [dataOut] = add_noise_transform(dataIn, noiseFactor, imgSize)
    % Ensure grayscale and resize
    img = imresize(dataIn, [imgSize imgSize]);
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    img = im2single(img); % Convert to [0,1] range
    
    % Add Gaussian Noise
    noise = noiseFactor * randn(size(img), 'single');
    noisyImg = img + noise;
    noisyImg = max(0, min(1, noisyImg)); % Clip
    
    % Return cell array {Input, Response} for trainNetwork
    dataOut = {noisyImg, img};
end