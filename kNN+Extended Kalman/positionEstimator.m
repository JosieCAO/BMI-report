%%% Team NAME : bls
%%% Team Members: Josephine Cao, Jiayu Liu, Xinyi Liu, Fangyuan Wang
%%% BMI Spring 2024 (Update 17th March 2024)

function [x, y, newModelParameters] = positionEstimator(testData, modelParameters)
% POSITIONESTIMATOR estimates the position (x, y) of a hand movement based on neural signals.
% Inputs:
%   test_data - The neural signals data for testing.
%   modelParameters - Struct containing the parameters of the model trained using training data.
% Outputs:
%   x, y - Estimated coordinates of hand movement.
%   newModelParameters - Updated model parameters after processing test_data.

    % Time window for processing signals
    dt = 20;
    % Determine the maximum time point in the given test data
    tmax = length(testData.spikes);
    % Calculate the minimum time point for the current window
    tmin = tmax - dt;
    
    % Get the number of neurons based on the size of the spikes matrix
    num_neurons = length(testData.spikes(:,1));
    % Determine if the current test data segment is within the initial 320ms window

    fla = (tmax-300)/dt;
    
    %% Estimate reaching angle
    % If the test data is within the initial 320ms, predict movement direction
    % otherwise, use the previously predicted direction.
    if fla <= 1
        % Sum spikes across all neurons to get the spike count for each neuron
        sum_spikes = sum(testData.spikes(:,50:320),2)';
        % Access the trained KNN model from the model parameters
        mdl = modelParameters.knnModel;
        % Predict the movement direction using the custom KNN predictor
        predict_angles = mode(customPredictKNN(mdl, sum_spikes));
        % Update the direction in the model parameters
        modelParameters(1).direction = predict_angles;
    else
        % Use the previously predicted direction if beyond initial 320ms
        predict_angles = modelParameters(1).direction;
    end
    
    %% Firing Rate Calculation
    % Initialize array for storing firing rates of all neurons
    store_spikerates = zeros(1,num_neurons);
    % Calculate firing rate for each neuron in the specified time window
    for i = 1:num_neurons
        sum_spikes = sum(testData.spikes(i,tmin:tmax));
        store_spikerates(i) = sum_spikes / dt; % Normalize by the window duration
    end
    
    %% Kalman Filter Update
    % Update the hand position estimates using the Kalman filter
    [x, y, newModelParameters] = ExtKalmanFilter(testData, store_spikerates, modelParameters, predict_angles, fla); 

end

function [xfin, yfin, newModelParameters] = ExtKalmanFilter(datainput, response, modelpar, direct, flag)
% KALMANFILTER updates the position estimate using a Kalman filter.
% Inputs:
%   datainput - Test data input containing neural signals.
%   response - The response or the firing rate of neurons.
%   modelpar - Current model parameters.
%   direct - Predicted direction of movement.
%   flag - Indicates if the position should be initialized (true) or updated (false).
% Outputs:
%   x_prediction, y_prediction - Updated position estimates.
%   newModelParameters - Updated model parameters including the Kalman filter state.

    % Access the Kalman filter parameters for the estimated direction
    filter = modelpar(direct).kalModel;
    dt = 20;
    % Initialize arrays for corrected state and covariance (unused in this context)
    xCorrected = [];
    PCorrected = [];
    
    % Extract the trained matrices from the Kalman filter model
    A = filter.A; % State transition matrix
    W = filter.W; % Process noise covariance matrix
    H = filter.H; % Measurement matrix
    Q = filter.Q; % Measurement noise covariance matrix

    % Initialize or retrieve the state and covariance matrix based on the flag
    if flag <= 1
        % For initial position, calculate relative to the center
        x = datainput.startHandPos(1,1); % Initial x position
        y = datainput.startHandPos(2,1); % Initial y position
        x0 = [(x - filter.center(1)), (y - filter.center(2)), 0, 0]'; % Initial state vector
        p0 = eye(length(A)); % Initial covariance matrix (identity matrix scaled by A's size)
    else
        % Use the previous state and covariance if not initializing
        x0 = filter.x0;
        p0 = filter.P0;
    end

    % Prediction step: predict the next state and covariance
    funs = @(x0)A*x0;
    funm = @(x0)H*x0;
    [xCorrected,PCorrected] = ekf(funs,x0,p0,funm,response,W,Q);
    

    % Convert corrected state back to original coordinates and output
    x_prediction = xCorrected(1) + filter.center(1); % Updated x position
    y_prediction = xCorrected(2) + filter.center(2); % Updated y position
    if flag > 14
        flag = 14;
    end

    pos = filter.poscent(:,flag);
    wei = 0.7;
    xfin = wei*x_prediction + (1-wei)*pos(1);
    yfin = wei*y_prediction + (1-wei)*pos(2);
    xCorrected(1) = xfin - filter.center(1);
    xCorrected(2) = yfin - filter.center(2);
    % Update the model parameters with the new state and covariance for future time steps
    filter.x0 = xCorrected;
    filter.P0 = PCorrected;
    modelpar(direct).kalModel = filter;
    newModelParameters = modelpar; % Pass along any unchanged parameters
    
end

% function [x_prediction, y_prediction, newModelParameters] = kalmanFilter(datainput, response, modelpar, direct, flag)
% % KALMANFILTER updates the position estimate using a Kalman filter.
% % Inputs:
% %   datainput - Test data input containing neural signals.
% %   response - The response or the firing rate of neurons.
% %   modelpar - Current model parameters.
% %   direct - Predicted direction of movement.
% %   flag - Indicates if the position should be initialized (true) or updated (false).
% % Outputs:
% %   x_prediction, y_prediction - Updated position estimates.
% %   newModelParameters - Updated model parameters including the Kalman filter state.
% 
%     % Access the Kalman filter parameters for the estimated direction
%     filter = modelpar(direct).kalModel;
% 
%     % Initialize arrays for corrected state and covariance (unused in this context)
%     xCorrected = [];
%     PCorrected = [];
%     
%     % Extract the trained matrices from the Kalman filter model
%     A = filter.A; % State transition matrix
%     W = filter.W; % Process noise covariance matrix
%     H = filter.H; % Measurement matrix
%     Q = filter.Q; % Measurement noise covariance matrix
% 
%     % Initialize or retrieve the state and covariance matrix based on the flag
%     if flag
%         % For initial position, calculate relative to the center
%         x = datainput.startHandPos(1,1); % Initial x position
%         y = datainput.startHandPos(2,1); % Initial y position
%         x0 = [(x - filter.center(1)), (y - filter.center(2)), 0, 0]'; % Initial state vector
%         p0 = eye(length(A)); % Initial covariance matrix (identity matrix scaled by A's size)
%     else
%         % Use the previous state and covariance if not initializing
%         x0 = filter.x0;
%         p0 = filter.P0;
%     end
% 
%     % Prediction step: predict the next state and covariance
%     xPred = A * x0; % Predicted state
%     PPred = A * p0 * A' + W; % Predicted covariance
% 
%     % Measurement update step: incorporate the new measurement
%     y = response' - H * xPred; % Measurement residual (difference between observed and predicted measurement)
%     S = H * PPred * H' + Q; % Residual covariance
% 
%     K = PPred * H' / S; % Kalman gain: weights the amount of the measurement update
%     xCorrected = xPred + K * y; % Corrected state estimate
%     PCorrected = (eye(size(K * H)) - K * H) * PPred; % Corrected covariance estimate
%     
%     % Update the model parameters with the new state and covariance for future time steps
%     filter.x0 = xCorrected;
%     filter.P0 = PCorrected;
%     modelpar(direct).kalModel = filter;
%     newModelParameters = modelpar; % Pass along any unchanged parameters
%     
%     % Convert corrected state back to original coordinates and output
%     x_prediction = xCorrected(1) + filter.center(1); % Updated x position
%     y_prediction = xCorrected(2) + filter.center(2); % Updated y position
% end

% Linear regression method. (not submitted)
% Use the start position if it is the beginning of the test data
% Otherwise, update the position using previous position + velocity * 20ms

% % Estimate velocity
% reach = modelParameters(direction).linear;
% velocity_x = total_firing_rate*reach(:,1);
% velocity_y = total_firing_rate*reach(:,2);
% reach = [];
% if length(test_data.spikes) <= 320  
%     xl = test_data.startHandPos(1);
%     yl = test_data.startHandPos(2);
%     fla = 1;
% else 
%     xl = test_data.decodedHandPos(1,length(test_data.decodedHandPos(1,:))) + velocity_x*(dt);
%     yl = test_data.decodedHandPos(2,length(test_data.decodedHandPos(2,:))) + velocity_y*(dt);
%     fla = 0;
% end
% % Update modelparameters
% newModelParameters.linear = modelParameters.linear;
% newModelParameters.knnModel = modelParameters.knnModel;
% newModelParameters.direction = direction;
% 
% % previous positions
% past_pos = test_data.decodedHandPos;
% % Measurement from KNN
% measurement = [x;y];
