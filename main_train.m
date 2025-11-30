% Medical MRI Denoising - Training Script
clear; clc; close all;
addpath('src');

dataDir = 'data/raw';
imgSize = 128;
noiseFactor = 0.3;
maxEpochs = 20;
batchSize = 16;
modelType = 'UNet'; % Options: 'CAE', 'UNet', 'ResNet'


disp('Loading and Splitting Data...');
[dsTrain, dsVal, dsTest] = load_data(dataDir, noiseFactor, imgSize);


inputSize = [imgSize imgSize 1];
switch modelType
    case 'CAE'
        lgraph = define_cae(inputSize);
    case 'UNet'
        lgraph = define_unet(inputSize);
    case 'ResNet'
        lgraph = define_resnet(inputSize);
    otherwise
        error('Unknown model type');
end


options = trainingOptions('adam', ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', batchSize, ...
    'InitialLearnRate', 1e-3, ...
    'ValidationData', dsVal, ...
    'ValidationFrequency', 50, ...
    'Plots', 'training-progress', ...
    'Verbose', false, ...
    'OutputNetwork', 'best-validation-loss');


disp(['Starting Training for ' modelType '...']);
net = trainNetwork(dsTrain, lgraph, options);

if ~exist('saved_models', 'dir')
    mkdir('saved_models');
end
save(['saved_models/' modelType '_Net.mat'], 'net');
disp('Training Complete. Model Saved.');