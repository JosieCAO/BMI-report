function YPred = customPredictKNN(model, XTest)
    % 自定义的KNN预测函数
    % 输入:
    % model - 由customFitKNN返回的KNN模型
    % XTest - 测试数据的特征，大小为m x n（m个样本，n个特征）
    % 输出:
    % YPred - 预测的标签，大小为m x 1
    
    numTestSamples = size(XTest, 1);
    YPred = zeros(numTestSamples, 1);
    
    for i = 1:numTestSamples
        distances = sqrt(sum((model.XTrain - XTest(i, :)).^2, 2));
        [~, sortedIndices] = sort(distances);
        kNearestNeighbors = model.YTrain(sortedIndices(1:model.K));
        YPred(i) = mode(kNearestNeighbors);
    end
end    
