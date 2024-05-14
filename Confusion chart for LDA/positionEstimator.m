function [predictedDirection, newModelParameters] = positionEstimator(past_current_trial, modelParameters)
    
    spike_counts = sum(past_current_trial.spikes(:, 1:320), 2)'; % 取前320ms的数据
    

    predictedDirection = predict(modelParameters.mdl, spike_counts);
    

    newModelParameters = modelParameters;
end
