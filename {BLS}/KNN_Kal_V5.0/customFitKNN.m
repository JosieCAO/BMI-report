function model = customFitKNN(XTrain, YTrain, K)
    % KNN training function, store KNN core matrix.
    % Input:
    % XTrain - Features of the training data, size m x n (m samples, n features)
    % YTrain - The label for the training data, size m x 1
    % K - The number of nearest neighbors
    % Output:
    % model - A structure that contains training data, labels, and K values
    
    model.XTrain = XTrain;
    model.YTrain = YTrain;
    model.K = K;
end


