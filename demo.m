clc
clear
close all

imds = imageDatastore("/home/home/pylib/dataset/DigitsData/", IncludeSubfolders=true, LabelSource="foldernames");
[imdsTrain,imdsValidation,imdsTest] = splitEachLabel(imds,0.7,0.15,"randomized");

inputSize = [28 28 1];
pixelRange = [-5 5];
imageAugmenter = imageDataAugmenter(RandXTranslation=pixelRange, RandYTranslation=pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain,DataAugmentation=imageAugmenter);

classes = categories(imdsTrain.Labels);
numClasses = numel(classes);

layers = [
    imageInputLayer(inputSize,Normalization="none")
    convolution2dLayer(5,20,Padding="same")
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,20,Padding="same")
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,20,Padding="same")
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer];

%Train
options.numEpochs = 10;
options.miniBatchSize = 128;
options.initialLearnRate = 0.01;
options.decay = 0;
options.momentum = 0.9;
options.imdsValidation = imdsValidation;

net = customTrainNetwork(augimdsTrain, layers, options);

% test
YTest = customClassify(net, imdsTest, options);
TTest = imdsTest.Labels;
accuracy = 100*mean(TTest == YTest)

figure
confusionchart(TTest,YTest)

