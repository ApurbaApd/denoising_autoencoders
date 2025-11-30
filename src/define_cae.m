function lgraph = define_cae(inputSize)
    layers = [
        imageInputLayer(inputSize, 'Name', 'input', 'Normalization', 'none')
        
        % Encoder
        convolution2dLayer(3, 32, 'Padding', 'same', 'Stride', 2, 'Name', 'enc1')
        reluLayer('Name', 'relu1')
        convolution2dLayer(3, 64, 'Padding', 'same', 'Stride', 2, 'Name', 'enc2')
        reluLayer('Name', 'relu2')
        
        % Decoder
        transposedConv2dLayer(3, 32, 'Cropping', 'same', 'Stride', 2, 'Name', 'dec1')
        reluLayer('Name', 'relu3')
        transposedConv2dLayer(3, 1, 'Cropping', 'same', 'Stride', 2, 'Name', 'dec2')
        
        % Output
        clippedReluLayer(1.0, 'Name', 'sigmoid_sim') % Simulates Sigmoid [0,1]
        regressionLayer('Name', 'mse_output')
    ];
    lgraph = layerGraph(layers);
end