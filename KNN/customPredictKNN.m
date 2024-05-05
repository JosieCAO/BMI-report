function YPred = customPredictKNN(model, XTest)
    % �Զ����KNNԤ�⺯��
    % ����:
    % model - ��customFitKNN���ص�KNNģ��
    % XTest - �������ݵ���������СΪm x n��m��������n��������
    % ���:
    % YPred - Ԥ��ı�ǩ����СΪm x 1
    
    numTestSamples = size(XTest, 1);
    YPred = zeros(numTestSamples, 1);
    
    for i = 1:numTestSamples
        distances = sqrt(sum((model.XTrain - XTest(i, :)).^2, 2));
        [~, sortedIndices] = sort(distances);
        kNearestNeighbors = model.YTrain(sortedIndices(1:model.K));
        YPred(i) = mode(kNearestNeighbors);
    end
end    
