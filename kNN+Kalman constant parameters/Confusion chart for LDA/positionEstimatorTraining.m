function modelParameters = positionEstimatorTraining(training_data)
    trials = size(training_data, 1);
    angle = size(training_data, 2);
    neurons = size(training_data(1,1).spikes, 1);

    spikes = [];
    direction = [];

    
    for a = 1:angle
        for t = 1:trials
            spike_count = sum(training_data(t, a).spikes(:, 1:320), 2);
            spikes = [spikes; spike_count'];
            direction = [direction; a];
        end
    end

    % LDA model
    mdl = fitcdiscr(spikes, direction, 'DiscrimType', 'linear');
    modelParameters.mdl = mdl;
end
