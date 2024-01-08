function net = customTrainNetwork(imds_train, layers, options)

net = dlnetwork(layers);

numEpochs = options.numEpochs;
miniBatchSize = options.miniBatchSize;
initialLearnRate = options.initialLearnRate;
decay = options.decay;
momentum = options.momentum;
imds_validation = options.imdsValidation;

mbq_train = minibatchqueue(imds_train,...
    MiniBatchSize = miniBatchSize,...
    MiniBatchFcn = @preprocessMiniBatch,...
    MiniBatchFormat = ["SSCB" ""], ...
    PartialMiniBatch = "discard");

mbq_validation = minibatchqueue(imds_validation, 1, ...
    MiniBatchSize=length(imds_validation.Labels), ...
    MiniBatchFcn=@preprocessMiniBatchPredictors, ...
    MiniBatchFormat="SSCB");
X_validation = next(mbq_validation);

velocity = [];

numObservationsTrain = numel(imds_train.Files);
numIterationsPerEpoch = floor(numObservationsTrain / miniBatchSize);
numIterations = numEpochs * numIterationsPerEpoch;

monitor = trainingProgressMonitor;

monitor.Metrics = ["TrainingAccuracy","ValidationAccuracy", ...
    "TrainingLoss","ValidationLoss"];
groupSubPlot(monitor,"Accuracy",["TrainingAccuracy","ValidationAccuracy"]);
groupSubPlot(monitor,"Loss",["TrainingLoss","ValidationLoss"]);

monitor.Info = ["LearningRate","TrainingLoss","TrainingAccuracy","ValidationLoss","ValidationAccuracy","Epoch","IterationsPerEpoch","Iteration","ExecutionEnvironment"];
monitor.XLabel = "Iteration";
monitor.Progress = 0;

executionEnvironment = "auto";
if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    updateInfo(monitor,ExecutionEnvironment="GPU");
else
    updateInfo(monitor,ExecutionEnvironment="CPU");
end

epoch = 0;
iteration = 0;

monitor.Status = "Running";

best_net = net;
best_validation_accuracy = 0;
% Loop over epochs.
while epoch < numEpochs && ~monitor.Stop
    epoch = epoch + 1;

    shuffle(mbq_train);

    % Loop over mini-batches.
    while hasdata(mbq_train) && ~monitor.Stop
        iteration = iteration + 1;

        [X_train,T_train] = next(mbq_train);

        % calc training loss
        [Training_loss,training_gradients,training_state] = dlfeval(@modelLoss,net,X_train,T_train);
        net.State = training_state;

        % update network
        learnRate = initialLearnRate/(1 + decay*iteration);
        [net,velocity] = sgdmupdate(net,training_gradients,velocity,learnRate,momentum);

        % calc training accuracy
        [~, Tdecode_train] = max(T_train);
        training_scores = predict(net,X_train);
        [~, Ydecode_train] = max(training_scores);
        Training_accuracy = 100*mean(Tdecode_train == Ydecode_train);

        % calc validation loss and accuracy.
        if iteration == 1 || ~hasdata(mbq_train)
            Ydecode_validation = customClassify(net, imds_validation, options);
            Tdecode_validation = imds_validation.Labels;
            Validation_accuracy = 100*mean(Tdecode_validation == Ydecode_validation);
            
            T_validation = onehotencode(Tdecode_validation,2)';
            Validation_loss = dlfeval(@modelLoss,net,X_validation,T_validation);

            recordMetrics(monitor,iteration, ...
                ValidationLoss = Validation_loss, ...
                ValidationAccuracy = Validation_accuracy);
            
            if best_validation_accuracy <= Validation_accuracy
                best_net = net;
                best_validation_accuracy = Validation_accuracy;
            end
        end

        % update monitor
        recordMetrics(monitor,iteration, ...
            TrainingLoss = Training_loss, ...
            TrainingAccuracy = Training_accuracy);
        
        updateInfo(monitor, ...
            LearningRate = learnRate, ...
            Epoch = string(epoch) + " of " + string(numEpochs), ...
            Iteration = string(iteration) + " of " + string(numIterations), ...
            IterationsPerEpoch = string(mod(iteration,numIterationsPerEpoch)) + " of " + string(numIterationsPerEpoch), ...
            TrainingLoss = string(Training_loss), ...
            TrainingAccuracy = string(Training_accuracy), ...
            ValidationLoss = string(Validation_loss), ...
            ValidationAccuracy = string(Validation_accuracy) + " (best: " + string(best_validation_accuracy) + ")");

        monitor.Progress = 100 * iteration/numIterations;
    end
end

if monitor.Stop == 1
    monitor.Status = "Training stopped";
else
    monitor.Status = "Training complete";
end

net = best_net;