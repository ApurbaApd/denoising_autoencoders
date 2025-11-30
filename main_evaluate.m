% Medical MRI Denoising - Evaluation Script
clear; clc; close all;
addpath('src');


imgSize = 128;
noiseFactor = 0.3;
dataDir = 'data/raw';

% Load ONLY the Test Set (re-creating split logic for consistency)
[~, ~, dsTest] = load_data(dataDir, noiseFactor, imgSize);

% Load Models
models = {'CAE', 'UNet'}; % Add ResNet if trained
results = struct();

% Evaluation Loop 
disp('Evaluating Models on Test Set...');

% Get one batch for visualization
dataBatch = read(dsTest);
noisyInput = dataBatch{1};
cleanTarget = dataBatch{2};

fprintf('%-10s | %-10s | %-10s | %-10s\n', 'Model', 'PSNR', 'SSIM', 'EPI');
fprintf('----\n');

for i = 1:length(models)
    mName = models{i};
    try
        load(['saved_models/' mName '_Net.mat'], 'net');
        
        % Inference
        reconstruction = predict(net, noisyInput);
        
        % Calculate Metrics
        [p, s, e] = calculate_metrics(cleanTarget, reconstruction);
        
        fprintf('%-10s | %-10.2f | %-10.4f | %-10.4f\n', mName, p, s, e);
        
        % Store for plotting
        results.(mName) = reconstruction;
        
    catch
        warning(['Model ' mName ' not found. Train it first.']);
    end
end


figure('Position', [100, 100, 1000, 400]);

subplot(1, length(models)+2, 1);
imshow(cleanTarget); title('Original');

subplot(1, length(models)+2, 2);
imshow(noisyInput); title('Noisy Input');

plotIdx = 3;
fields = fieldnames(results);
for i = 1:numel(fields)
    subplot(1, length(models)+2, plotIdx);
    imshow(results.(fields{i}));
    title([fields{i} ' Result']);
    plotIdx = plotIdx + 1;
end