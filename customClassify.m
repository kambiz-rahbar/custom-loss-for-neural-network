function predicted_labels = customClassify(net, imdsTest, options)

miniBatchSize = options.miniBatchSize;
classes = categories(imdsTest.Labels);

numOutputs = 1;

mbqTest = minibatchqueue(imdsTest,numOutputs, ...
    MiniBatchSize=miniBatchSize, ...
    MiniBatchFcn=@preprocessMiniBatchPredictors, ...
    MiniBatchFormat="SSCB");

predicted_labels = modelPredictions(net,mbqTest,classes);
