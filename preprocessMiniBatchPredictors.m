function X = preprocessMiniBatchPredictors(dataX)
    X = cat(4,dataX{1:end});
end