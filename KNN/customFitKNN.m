function model = customFitKNN(XTrain, YTrain, K)
    % 自定义的KNN训练函数
    % 输入:
    % XTrain - 训练数据的特征，大小为m x n（m个样本，n个特征）
    % YTrain - 训练数据的标签，大小为m x 1
    % K - 最近邻的数量
    % 输出:
    % model - 包含训练数据、标签和K值的结构体
    
    model.XTrain = XTrain;
    model.YTrain = YTrain;
    model.K = K;
    end


