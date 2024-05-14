
    load('monkeydata_training.mat')

    rng(2013);
    ix = randperm(length(trial));
    trainingData = trial(ix(1:50), :);
    testData = trial(ix(51:end), :);

 
    [X_train, y_train] = extract_features_labels(trainingData);
    [X_test, y_test] = extract_features_labels(testData);

    % Evaluate validation errors for different K values
    max_k = 80;
    % max_k = 400;
    errors = evaluate_knn(X_train, y_train, X_test, y_test, max_k);
    k_values = 1:max_k;  % This ensures k_values and errors have the same length

    % Plot validation error
    figure;
    plot(k_values, errors, 'o-', 'Color', 'blue');
    
    xlabel('k Value');
    ylabel('Validation Error');

    % Customizing the axes to remove top and right borders
    ax = gca;  % Get current axes
    ax.Box = 'off';
    ax.XAxisLocation = 'origin';
    ax.YAxisLocation = 'origin';
    grid on;


function [features, labels] = extract_features_labels(trials)
    [n, k] = size(trials);  % n: number of trials, k: number of reaching angles
    features = [];
    labels = [];
    for i = 1:n
        for j = 1:k
            spikes = trials(i, j).spikes;
            spike_counts = sum(spikes(:, 1:320), 2)';  % Sum spikes in the first 320 ms
            features = [features; spike_counts];
            labels = [labels; j];
        end
    end
end

function errors = evaluate_knn(X_train, y_train, X_test, y_test, max_k)
    errors = zeros(1, max_k);
    for k = 1:max_k
        model = customFitKNN(X_train, y_train, k);
        y_pred = customPredictKNN(model, X_test);
        accuracy = sum(y_pred == y_test) / length(y_test);
        errors(k) = 1 - accuracy;  % Each k value from 1 to max_k will add an error value
    end
end
