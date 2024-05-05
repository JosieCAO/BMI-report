function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)

%% estimate reaching angle
% % �����ƶ�����
% % ����������ݵ�ʱ�䳤��С�ڻ����320ms����ʹ��KNNģ�ͺ͵�ǰ���񾭻������Ԥ���ֱ۵��ƶ�����
% % ���ʱ�䳤�ȴ���320ms����ʹ��֮ǰԤ��õ��ķ���.

% % �ڲ������ݵ����ڽ׶Σ�ǰ320ms����������KNNģ�ͽ���Ԥ�⣻
% % ��֮��������跽�򲻱䣬����ʹ�������Ԥ������
% % ���ַ�����һ������ԭ���ǣ��ֱ۵Ĵ����ƶ��������ƶ���ʼʱ���Ѿ�ȷ�������ڶ�ʱ���ڲ��ᷢ�������仯��
spike_count = zeros(98, 1);
    
    if length(test_data.spikes) <= 320
        for i = 1:98
            total_spikes = length(find(test_data.spikes(i, 1:320) == 1));
            spike_count(i) = total_spikes;
        end
        % ʹ���Զ����Ԥ�⺯��������predict
        direction = mode(customPredictKNN(modelParameters.knnModel, spike_count'));
    else
        direction = modelParameters.direction;
    end

%% knn prediction
% % takes x,y position and vz,vy (velocity of reaching angles) in a suvat equation

% % ����ȷ��һ�����ڼ��㷢���ʵ�ʱ�䴰�ڣ�������ڴӲ������ݵ��񾭻��¼��ĩβ��ǰ��20���루ms��
tmin = length(test_data.spikes)-20;
tmax = length(test_data.spikes);

% firing rate calculated ���ڵ�ǰʱ�䴰���ڵ��񾭻���ݼ��㷢����
total_firing_rate = zeros(98,1);
for i = 1:98
    total_spikes = length(find(test_data.spikes(i,tmin:tmax)==1));
    total_firing_rate(i) = total_spikes/(20*0.001); % divided by 20ms time window
end

% estimate velocity �����ٶȣ�ʹ��ѵ���׶εõ���linear reggression�͵�ǰ�ķ������������ֱ���x��y�����ϵ��ٶȡ�
velocity_x = total_firing_rate'*modelParameters.beta(direction).reach_angle(:,1);
velocity_y = total_firing_rate'*modelParameters.beta(direction).reach_angle(:,2);

%% update parameters
%%�����ֱ�λ�ã�����ǲ������ݵĿ�ʼ�׶Σ���ʹ����ʼλ�ã�
if length(test_data.spikes) <= 320
    x = test_data.startHandPos(1);
    y = test_data.startHandPos(2);
else
    %%���򣬻���֮ǰ��λ�ú͹��Ƶ��ٶȸ����ֱ۵�λ��:previous position + velocity * 20ms (to get position) -> suvat
    x = test_data.decodedHandPos(1,length(test_data.decodedHandPos(1,:))) + velocity_x*(20*0.001);
    y = test_data.decodedHandPos(2,length(test_data.decodedHandPos(2,:))) + velocity_y*(20*0.001);
end

%%����ģ�Ͳ���������ǰ���򱣴�Ϊģ�Ͳ�����һ���֣��Ա�����һ��Ԥ��ʱʹ�á�
newModelParameters.beta = modelParameters.beta;
newModelParameters.knnModel = modelParameters.knnModel;
newModelParameters.direction = direction;

end
