function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)

%% estimate reaching angle
% % 估计移动方向：
% % 如果测试数据的时间长度小于或等于320ms，则使用KNN模型和当前的神经活动数据来预测手臂的移动方向；
% % 如果时间长度大于320ms，则使用之前预测得到的方向.

% % 在测试数据的早期阶段（前320ms），它利用KNN模型进行预测；
% % 在之后，它则假设方向不变，继续使用最初的预测结果。
% % 这种方法的一个可能原因是，手臂的大致移动方向在移动开始时就已经确定，且在短时间内不会发生显著变化。
spike_count = zeros(98, 1);
    
    if length(test_data.spikes) <= 320
        for i = 1:98
            total_spikes = length(find(test_data.spikes(i, 1:320) == 1));
            spike_count(i) = total_spikes;
        end
        % 使用自定义的预测函数而不是predict
        direction = mode(customPredictKNN(modelParameters.knnModel, spike_count'));
    else
        direction = modelParameters.direction;
    end

%% knn prediction
% % takes x,y position and vz,vy (velocity of reaching angles) in a suvat equation

% % 首先确定一个用于计算发射率的时间窗口，这个窗口从测试数据的神经活动记录的末尾向前推20毫秒（ms）
tmin = length(test_data.spikes)-20;
tmax = length(test_data.spikes);

% firing rate calculated 基于当前时间窗口内的神经活动数据计算发射率
total_firing_rate = zeros(98,1);
for i = 1:98
    total_spikes = length(find(test_data.spikes(i,tmin:tmax)==1));
    total_firing_rate(i) = total_spikes/(20*0.001); % divided by 20ms time window
end

% estimate velocity 估计速度：使用训练阶段得到的linear reggression和当前的发射率来估计手臂在x和y方向上的速度。
velocity_x = total_firing_rate'*modelParameters.beta(direction).reach_angle(:,1);
velocity_y = total_firing_rate'*modelParameters.beta(direction).reach_angle(:,2);

%% update parameters
%%更新手臂位置：如果是测试数据的开始阶段，则使用起始位置；
if length(test_data.spikes) <= 320
    x = test_data.startHandPos(1);
    y = test_data.startHandPos(2);
else
    %%否则，基于之前的位置和估计的速度更新手臂的位置:previous position + velocity * 20ms (to get position) -> suvat
    x = test_data.decodedHandPos(1,length(test_data.decodedHandPos(1,:))) + velocity_x*(20*0.001);
    y = test_data.decodedHandPos(2,length(test_data.decodedHandPos(2,:))) + velocity_y*(20*0.001);
end

%%更新模型参数：将当前方向保存为模型参数的一部分，以便在下一次预测时使用。
newModelParameters.beta = modelParameters.beta;
newModelParameters.knnModel = modelParameters.knnModel;
newModelParameters.direction = direction;

end
