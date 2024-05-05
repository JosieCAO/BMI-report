 function [modelParameters] = positionEstimatorTraining(training_data)

%% Find firing rates计算发射率：遍历每个试验和神经元，计算指定时间窗内的神经元发射率。
spike_rate = [];
firing_rates = [];
temp_vx = [];
temp_vy = [];
trainingData = struct([]);
velocity = struct([]);
dt = 20; % bin size of 20

for dirn = 1:8
    for neuron = 1:98
        for n = 1:length(training_data)
            for t = 300:dt:550-dt
                
                % find the firing rates of one neural unit for one trial
                total_spikes = length(find(training_data(n,dirn).spikes(neuron,t:t+dt)==1));
                spike_rate = cat(2, spike_rate, total_spikes/(dt*0.001));
                
                % find the velocity of the hand movement计算手部移动速度
                % (needs calculating just once for each trial)对于每次试验的每个方向，计算手部在x和y方向上的速度（利用位置变化除以时间）
                if neuron ==1
                    x1 = training_data(n,dirn).handPos(1,t);
                    x2 = training_data(n,dirn).handPos(1,t+dt);
                    y1 = training_data(n,dirn).handPos(2,t);
                    y2 = training_data(n,dirn).handPos(2,t+dt);
                    
                    vx = (x2 - x1) / (dt*0.001);
                    vy = (y2 - y1) / (dt*0.001);
                    temp_vx = cat(2, temp_vx, vx);
                    temp_vy = cat(2, temp_vy, vy);
                end
            end
            % store firing rates and concat for each neuron+trial
            firing_rates = cat(2, firing_rates, spike_rate);
            spike_rate = [];
        end
        trainingData(neuron,dirn).firing_rates = firing_rates;
        velocity(dirn).x = temp_vx;
        velocity(dirn).y = temp_vy;
       
        firing_rates = [];
    end
    temp_vx = [];
    temp_vy = [];
end

%% Linear Regression线性回归：
% used to predict velocity使用神经元的发射率作为特征和手部速度作为目标，对每个方向单独训练一个线性回归模型，预测手部的速度。
beta = struct([]);

for dirn=1:8
    vel = [velocity(dirn).x; velocity(dirn).y];
    total_firing_rate = [];
    for n=1:98
    total_firing_rate = cat(1, total_firing_rate, trainingData(n,dirn).firing_rates);
    end 
    beta(dirn).reach_angle = lsqminnorm(total_firing_rate',vel');
end

%% KNN Classifier
    % 计算每个试验前320ms内的神经元总脉冲数，用这些数据训练一个KNN模型，预测手臂移动的方向。
    
    spikes = [];
    reach_angle = [];
    spike_count = zeros(length(training_data), 98);
    
    for dirn = 1:8
        for neuron = 1:98
            for n = 1:length(training_data)
                total_spikes = length(find(training_data(n,dirn).spikes(neuron, 1:320) == 1));
                spike_count(n, neuron) = total_spikes;
            end
        end
        spikes = cat(1, spikes, spike_count);
        reaching_angle(1:length(training_data)) = dirn;
        reach_angle = cat(2, reach_angle, reaching_angle); 
    end

    K = 5; % 选择K值，即最近邻数量
    customKNNModel = customFitKNN(spikes, reach_angle, K);
    
    % 存储训练好的模型到modelParameters中
    modelParameters = struct('beta', beta, 'knnModel', customKNNModel);
    
    
end

% function model = customFitKNN(XTrain, YTrain, K)
%     % 自定义的KNN训练函数
%     % 输入:
%     % XTrain - 训练数据的特征，大小为m x n（m个样本，n个特征）
%     % YTrain - 训练数据的标签，大小为m x 1
%     % K - 最近邻的数量
%     % 输出:
%     % model - 包含训练数据、标签和K值的结构体
%     
%     model.XTrain = XTrain;
%     model.YTrain = YTrain;
%     model.K = K;
% end
% 
% function YPred = customPredictKNN(model, XTest)
%     % 自定义的KNN预测函数
%     % 输入:
%     % model - 由customFitKNN返回的KNN模型
%     % XTest - 测试数据的特征，大小为m x n（m个样本，n个特征）
%     % 输出:
%     % YPred - 预测的标签，大小为m x 1
%     
%     numTestSamples = size(XTest, 1);
%     YPred = zeros(numTestSamples, 1);
%     
%     for i = 1:numTestSamples
%         distances = sqrt(sum((model.XTrain - XTest(i, :)).^2, 2));
%         [~, sortedIndices] = sort(distances);
%         kNearestNeighbors = model.YTrain(sortedIndices(1:model.K));
%         YPred(i) = mode(kNearestNeighbors);
%     end
% end    

    
