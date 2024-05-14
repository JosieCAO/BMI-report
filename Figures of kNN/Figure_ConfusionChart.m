
    load monkeydata_training.mat

    rng(2013);
    ix = randperm(length(trial));
    trainingData = trial(ix(1:50), :);
    testData = trial(ix(51:end), :);


    modelParameters = positionEstimatorTraining(trainingData);

    actualDirections = [];
    predictedDirections = [];

    % knn classification
    for tr = 1:size(testData, 1)
        for direc = 1:8
            past_current_trial.spikes = testData(tr, direc).spikes;
            past_current_trial.startHandPos = testData(tr, direc).handPos(1:2, 1);

            [predictedDirection, newModelParameters] = positionEstimator(past_current_trial, modelParameters);

            actualDirections = [actualDirections; direc];
            predictedDirections = [predictedDirections; predictedDirection];
        end
    end

    % compute error and F1
    confMat = confusionmat(actualDirections, predictedDirections);
    numClasses = size(confMat, 1);
    precision = zeros(numClasses, 1);
    recall = zeros(numClasses, 1);
    f1Scores = zeros(numClasses, 1);

    for i = 1:numClasses
        TP = confMat(i, i);
        FP = sum(confMat(:, i)) - TP;
        FN = sum(confMat(i, :)) - TP;
        precision(i) = TP / (TP + FP);
        recall(i) = TP / (TP + FN);
        f1Scores(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
    end

    meanF1Score = mean(f1Scores);
    accuracy = sum(diag(confMat)) / sum(confMat, 'all');  

    figure;
    confusionchart(confMat);
    title(sprintf('Confusion Chart for kNN Classifier\nAccuracy: %.2f%%, Average F1 Score: %.2f', accuracy * 100, meanF1Score));

