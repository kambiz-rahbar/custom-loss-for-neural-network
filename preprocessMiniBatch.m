function [X,T] = preprocessMiniBatch(dataX,dataT)
    X = preprocessMiniBatchPredictors(dataX);
    T = cat(2,dataT{1:end}); % Extract label data from cell and concatenate.
    T = onehotencode(T,1);
end

