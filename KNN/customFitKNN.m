function model = customFitKNN(XTrain, YTrain, K)
    % �Զ����KNNѵ������
    % ����:
    % XTrain - ѵ�����ݵ���������СΪm x n��m��������n��������
    % YTrain - ѵ�����ݵı�ǩ����СΪm x 1
    % K - ����ڵ�����
    % ���:
    % model - ����ѵ�����ݡ���ǩ��Kֵ�Ľṹ��
    
    model.XTrain = XTrain;
    model.YTrain = YTrain;
    model.K = K;
    end


