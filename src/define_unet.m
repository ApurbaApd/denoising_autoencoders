function lgraph = define_unet(inputSize)
    lgraph = layerGraph();

    % Encoder
    tempLayers = [
        imageInputLayer(inputSize, 'Name', 'input', 'Normalization', 'none')
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv1')
        reluLayer('Name', 'relu1')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
        convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv2')
        reluLayer('Name', 'relu2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
    ];
    lgraph = addLayers(lgraph, tempLayers);

    % Bottleneck
    tempLayers = [
        convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'bottleneck')
        reluLayer('Name', 'relu_bot')
        transposedConv2dLayer(2, 128, 'Stride', 2, 'Name', 'up1')
    ];
    lgraph = addLayers(lgraph, tempLayers);

    % Decoder 1 (Concatenation)
    tempLayers = [
        concatenationLayer(3, 2, 'Name', 'concat1') % Skip connection
        convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'dec_conv1')
        reluLayer('Name', 'dec_relu1')
        transposedConv2dLayer(2, 64, 'Stride', 2, 'Name', 'up2')
    ];
    lgraph = addLayers(lgraph, tempLayers);

    % Decoder 2
    tempLayers = [
        concatenationLayer(3, 2, 'Name', 'concat2') % Skip connection
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'dec_conv2')
        reluLayer('Name', 'dec_relu2')
        convolution2dLayer(1, 1, 'Name', 'final_conv')
        regressionLayer('Name', 'output')
    ];
    lgraph = addLayers(lgraph, tempLayers);

    % Connect Layers
    lgraph = connectLayers(lgraph, 'pool2', 'bottleneck');
    lgraph = connectLayers(lgraph, 'up1', 'concat1/in1');
    lgraph = connectLayers(lgraph, 'relu2', 'concat1/in2'); % Skip 1
    lgraph = connectLayers(lgraph, 'up2', 'concat2/in1');
    lgraph = connectLayers(lgraph, 'relu1', 'concat2/in2'); % Skip 2
end