function YPred = customPredictKNN(model, XTest)
    % Custom KNN prediction function
    % Input:
    % model - The KNN model returned by customFitKNN
    % XTest - Features of the test data, size m x n (m samples, n features)
    % Output:
    % YPred - Predicted label with size m x 1
    
    numTestSamples = size(XTest, 1);
    YPred = zeros(numTestSamples, 1);
    
    for i = 1:numTestSamples
        distances = sqrt(sum((model.XTrain - XTest(i, :)).^2, 2));
        [~, sortedIndices] = sort(distances);
        kNearestNeighbors = model.YTrain(sortedIndices(1:model.K));
        YPred(i) = mode(kNearestNeighbors);
    end
end    
