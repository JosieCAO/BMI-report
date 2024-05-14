function [predictedDirection, newModelParameters] = positionEstimator(past_current_trial, modelParameters)
    
    spike_counts = sum(past_current_trial.spikes(:, 1:320), 2)';
    modes = modelParameters.knnModel;
    predictedDirection = mode(customPredictKNN(modes, spike_counts));
    
    newModelParameters = modelParameters;
end
