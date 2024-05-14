%%% Team NAME : bls
%%% Team Members: Josephine Cao, Jiayu Liu, Xinyi Liu, Fangyuan Wang
%%% BMI Spring 2024 (Update 17th March 2024)
% training_data = trial;
function [modelParameters] = positionEstimatorTraining(trainingDate)
% POSITIONESTIMATORTRAINING Trains the model parameters for estimating hand position.
% Input:
%   training_data - Struct containing the neural and hand position data for training.
% Output:
%   modelParameters - Struct containing the trained model parameters.

%% Initialize variables for data processing and feature extraction
store_sr = []; % Temporary storage for spike rates
spikerates = []; % Collection of firing rates across all trials and neurons
Ave_sr = []; % Aggregated firing rates for feature vector
store_vx = []; % Temporary storage for velocity in x-direction
store_vy = []; % Temporary storage for velocity in y-direction
disp_x = []; % Displacement in x-direction
disp_y = []; % Displacement in y-direction
vel_x = []; % Velocity in x-direction
vel_y = []; % Velocity in y-direction
pos_x = []; % Position in x-direction
pos_y = []; % Position in y-direction
poscent = {};
temx = [];
temy = [];
temposx = [];
temposy = [];
trainingData = {}; % Processed training data for model training
kinematics = {}; % Kinematic features extracted from training data
centers = {}; % Mean position for normalization purposes
windowSize = 20; % Time bin size for spike rate calculation
current_length = []; % To keep track of data lengths

%% Format data length and pre-process
% Determine the maximum length to standardize the length of neural signals
t_max = 0;
for i = 1:size(trainingDate, 1)
    for j = 1:size(trainingDate, 2)
        leng = size(trainingDate(i,j).spikes(1, :),2);
        current_length = cat(2, current_length, leng);
    end
end
t_max = floor((mean(current_length)-50)/windowSize)*windowSize;

% Pad the data with zeros to make all trials of minimun equal length
for i = 1:size(trainingDate, 1)
    for j = 1:size(trainingDate, 2)
        n_neurons = size(trainingDate(i,j).spikes, 1);
        for k = 1:n_neurons
            current_length = size(trainingDate(i,j).spikes(k, :), 2);
            if current_length < t_max
                trainingDate(i,j).spikes(k, (current_length+1):t_max) = 0;
                trainingDate(i,j).handPos(:,(current_length+1):t_max) = repmat(trainingDate(i,j).handPos(:,1), 1, (t_max-current_length)); 
            end
        end
    end
end

%% Feature Extraction
% Extracts spike rates and kinematic features (velocity and position) for model training
for angle = 1:8
    % Loop over each direction
    for neuron = 1:98
        % Loop over each neuron
        for n = 1:length(trainingDate)
            % Loop over each trial
            for t = 300:windowSize:t_max-windowSize
                % Calculate spike rates for each neuron within each time bin
                sum_spikes = sum(trainingDate(n,angle).spikes(neuron,t:t+windowSize));
                store_sr = cat(2, store_sr, sum_spikes/windowSize);
                
                % Calculate velocities and positions if first neuron (done once per trial)
                if neuron == 1
                    x1 = trainingDate(n,angle).handPos(1,t);
                    x2 = trainingDate(n,angle).handPos(1,t+windowSize);
                    y1 = trainingDate(n,angle).handPos(2,t);
                    y2 = trainingDate(n,angle).handPos(2,t+windowSize);
                    
                    velocityX = (x2 - x1) / windowSize;
                    velocityY = (y2 - y1) / windowSize;
    
                    % Store displacement and velocities
                    disp_x = cat(2,disp_x, x1);
                    disp_y = cat(2,disp_y, y1);
                    store_vx = cat(2, store_vx, velocityX);
                    store_vy = cat(2, store_vy, velocityY);

                    ntwin = floor((t-300)/windowSize)+1;
                    temx = cat(2,temx,mean(trainingDate(n,angle).handPos(1,t:t+windowSize)));
                    temy = cat(2,temy,mean(trainingDate(n,angle).handPos(2,t:t+windowSize)));

                end
            end
            
            % Aggregate firing rates for feature extraction
            spikerates = cat(1, spikerates, store_sr);
            if neuron == 1
                vel_x = cat(1, vel_x, store_vx);
                vel_y = cat(1, vel_y, store_vy);
                pos_x = cat(1, pos_x, disp_x);
                pos_y = cat(1, pos_y, disp_y); 
                temposx = cat(1,temposx,temx);
                temposy = cat(1,temposy,temy);
                temx = [];
                temy = [];

            end
            
            % Reset temporary variables for next iteration
            store_sr = [];
            store_vx = [];
            store_vy = [];
            disp_y = [];
            disp_x = [];
        end
        % Compute average firing rates across all trials for
        Ave_sr = cat(1, Ave_sr, mean(spikerates));
        % Reset firing_rate
        spikerates = [];
       
    end

    % Store processed data for each direction
    trainingData{angle} = Ave_sr;
    Ave_sr = [];
    
    % Calculate and store kinematic data and center positions for each direction
    kinematics{angle} = [mean(pos_x) - mean(pos_x,"all"); mean(pos_y) - mean(pos_y,"all"); mean(vel_x); mean(vel_y)]';
    centers{angle} = [mean(pos_x,"all"), mean(pos_y,"all")];

    poscent{angle} = [mean(temposx,1);  mean(temposy,1)];
    temposy = [];
    temposx = [];
    % Reset kinematic variables for the next direction
    vel_x = [];
    vel_y = [];
    pos_x = [];
    pos_y = [];

end

%% KNN Classifier Training
    % Calculate total number of spikes for each neuron within the first 320ms of each trial
    features = [];
    labels = [];
    % tem = [];
    spikeCounts = zeros(length(trainingDate), 98);
    for angle = 1:8
        for neuron = 1:98
            for n = 1:length(trainingDate)
                % tem = reshape(training_data(n,dirn).spikes(neuron, 200:320),20,[]);
                sum_spikes = sum(trainingDate(n,angle).spikes(neuron, 50:320));
                spikeCounts(n, neuron) = sum_spikes;
            end
        end
        features = cat(1, features, spikeCounts); % Compile spike count data
        Lable_angles = repmat(angle, length(trainingDate), 1); % Label data with direction
        labels = cat(1, labels, Lable_angles); 
    end

    K = 10; % Number of nearest neighbors for KNN
    customKNNModel = customFitKNN(features, labels, K); % Train KNN model

%% Kalman Filter Training
    kalfilter = extended_kalman_fit(kinematics,trainingData,centers,poscent); % Train Kalman filter using kinematic data
    % Store the trained model in modelParameters
    modelParameters = struct('knnModel', customKNNModel, 'kalModel', kalfilter); % 'linear', beta,
    % plot kal transition matrix A
    % plotParameters(kalfilter,8)
end

function plotParameters(network,num)
    % network is a struct containing parameters for 8 layers of a neural network

    % Plot parameters as heat maps
    figure(1);
    for i = 1:num
        % Weight matrix
        subplot(1,num, i);
        w = normalize(network{i}.H);
        imagesc(w);
        xlabel('[x,y,vx,vy]')
        ylabel('Neurons')
        colorbar;
        title(['Direction ', num2str(i), ' Weights']);
    end

    sgtitle('Neuron Transition Weights');
end

function [filter] = extended_kalman_fit(train,response,cent,pos)
    % KALMAN_FIT Trains Extended Kalman filters for each direction based on the given training data.
    % Algorithm based on [W. Wu, etc] paper "Neural Decoding of Cursor Motion
    % Using a Kalman Filter".
    % Assuming Kalman parameters A, Q, W, H remain unchanged during time
    % updates. Therefore, parameters can be trained based on the average
    % kinematics of the training data.

    % Inputs:
    %   train - Cell array where each cell contains kinematic data (position, velocity) for a specific direction.
    %   response - Cell array where each cell contains the corresponding neural firing data for each direction.
    %   cent - Cell array containing the mean position for the kinematic data in each direction.
    % Outputs:
    %   filter - Struct array containing the trained Kalman filter parameters for each direction.

    % Train the Kalman filter for each of the 8 directions
    for dirn = 1:8
        spike_train = response{dirn}; % Extract spike train data for the current direction
        kim = train{dirn}; % Extract kinematic data for the current direction

        % Compute the variables required for filtering.
        % Preparing data for least squares solution
        X2 = kim(2:end, :)'; % Kinematics at time t+1
        X1 = kim(1:end-1, :)'; % Kinematics at time t

        % Compute the state transition matrix A using least squares
        % This matrix predicts the next state based on the current state

        A = X2*X1'*inv((X1*X1')); 

        % Compute the process noise covariance matrix W
        % This matrix represents the uncertainty in the model prediction
        W = ((X2 - A*X1)*((X2 - A*X1)')); 

        % The least-squares-optimal transformation from kinematics to neural firing rate
        X = kim'; % Kinematic data
        Z = spike_train; % Neural firing data

        % Compute the observation matrix H using least squares
        % This matrix relates the state to the observed neural firing rates
        H = Z*X'*inv((X*X')); 

        % Compute the observation noise covariance matrix Q
        % This matrix represents the uncertainty in the measurements (spike counts)
        Q = ((Z - H*X)*((Z - H*X)')); 
        
        % Regularization to avoid singular matrices
        epsilon = 1e-4; % Small regularization factor
        % epsilon = '1' - assume perfect linearity to minimize the random position error.

        W = W + epsilon * eye(size(W)); % Regularize process noise covariance
        Q = Q + epsilon * eye(size(Q)); % Regularize measurement noise covariance

        % Store the computed matrices in the filter struct for the current direction
        filter{dirn}.Q = Q;
        filter{dirn}.H = H;
        filter{dirn}.A = A;
        filter{dirn}.W = W;
        % Store the mean position for the current direction
        filter{dirn}.center = cent{dirn};
        filter{dirn}.poscent = pos{dirn};
    end

end

% function [filter] = kalman_fit(train,response,cent)
%     % KALMAN_FIT Trains Kalman filters for each direction based on the given training data.
%     % Algorithm based on [W. Wu, etc] paper "Neural Decoding of Cursor Motion
%     % Using a Kalman Filter".
%     % Assuming Kalman parameters A, Q, W, H remain unchanged during time
%     % updates. Therefore, parameters can be trained based on the average
%     % kinematics of the training data.
% 
%     % Inputs:
%     %   train - Cell array where each cell contains kinematic data (position, velocity) for a specific direction.
%     %   response - Cell array where each cell contains the corresponding neural firing data for each direction.
%     %   cent - Cell array containing the mean position for the kinematic data in each direction.
%     % Outputs:
%     %   filter - Struct array containing the trained Kalman filter parameters for each direction.
% 
%     % Train the Kalman filter for each of the 8 directions
%     for dirn = 1:8
%         spike_train = response{dirn}; % Extract spike train data for the current direction
%         kim = train{dirn}; % Extract kinematic data for the current direction
% 
%         % Compute the variables required for filtering.
%         % Preparing data for least squares solution
%         X2 = kim(2:end, :)'; % Kinematics at time t+1
%         X1 = kim(1:end-1, :)'; % Kinematics at time t
% 
%         % Compute the state transition matrix A using least squares
%         % This matrix predicts the next state based on the current state
%         A = X2*X1'*inv((X1*X1')); 
% 
%         % Compute the process noise covariance matrix W
%         % This matrix represents the uncertainty in the model prediction
%         W = ((X2 - A*X1)*((X2 - A*X1)')); 
% 
%         % The least-squares-optimal transformation from kinematics to neural firing rate
%         X = kim'; % Kinematic data
%         Z = spike_train; % Neural firing data
% 
%         % Compute the observation matrix H using least squares
%         % This matrix relates the state to the observed neural firing rates
%         H = Z*X'*inv((X*X')); 
% 
%         % Compute the observation noise covariance matrix Q
%         % This matrix represents the uncertainty in the measurements (spike counts)
%         Q = ((Z - H*X)*((Z - H*X)')); 
%         
%         % Regularization to avoid singular matrices
%         epsilon = 1; % Small regularization factor
%         % epsilon = '1' - assume perfect linearity to minimize the random position error.
% 
%         W = W + epsilon * eye(size(W)); % Regularize process noise covariance
%         Q = Q + epsilon * eye(size(Q)); % Regularize measurement noise covariance
% 
%         % Store the computed matrices in the filter struct for the current direction
%         filter{dirn}.Q = Q;
%         filter{dirn}.H = H;
%         filter{dirn}.A = A;
%         filter{dirn}.W = W;
%         % Store the mean position for the current direction
%         filter{dirn}.center = cent{dirn};
%     end
% 
% end


% %% Linear Regression
%     % Train linear regression models for velocity prediction using spiking data
%     
% beta = struct([]);
% 
% for dirn=1:8
%     vel = kinematics{dirn}(:,3:4);
% 
%     spiketrain = trainingData{dirn};
%     beta{dirn} = lsqminnorm(spiketrain',vel);
% end