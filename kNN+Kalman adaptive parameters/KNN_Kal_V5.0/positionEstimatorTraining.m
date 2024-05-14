%%% Team NAME : bls
%%% Team Members: Josephine Cao, Jiayu Liu, Xinyi Liu, Fangyuan Wang
%%% BMI Spring 2024 (Update 17th March 2024)

function [modelParameters] = positionEstimatorTraining(training_data)
% POSITIONESTIMATORTRAINING Trains the model parameters for estimating hand position.
% Input:
%   training_data - Struct containing the neural and hand position data for training.
% Output:
%   modelParameters - Struct containing the trained model parameters.

%% Initialize variables for data processing and feature extraction
ntrial = size(training_data, 1);
ndire = size(training_data, 2);
nneurons = size(training_data(1,1).spikes, 1); % Fixed neuron size
dt = 20; % Time bin size for spike rate calculation
spike_rate = []; % Temporary storage for spike rates
firing_rates = []; % Collection of firing rates across all trials and neurons
total_firing_rates = []; % Aggregated firing rates for feature vector
temp_vx = []; % Temporary storage for velocity in x-direction
temp_vy = []; % Temporary storage for velocity in y-direction
temp_x = []; % Displacement in x-direction
temp_y = []; % Displacement in y-direction
trainingData = {}; % Processed training data for model training
kinematics = {}; % Kinematic features extracted from training data
trialid = zeros([1,ntrial]);
centers = {}; % Mean position for normalization purposes
current_length = []; % To keep track of data lengths
time_window = zeros([1,ndire]);


%% Format data length and pre-process
% Determine the maximum length to standardize the length of neural signals

for j = 1:ndire
    for i = 1:ntrial
        leng = size(training_data(i,j).spikes(1, :),2);
        current_length = [current_length, leng];
    end
end
t_max = floor(max(current_length)/dt)*dt;
twin = ((t_max-dt-300)/dt);
win = twin;
% Pad the data with zeros to make all trials of minimun equal length
for j = 1:ndire
    for i = 1:ntrial
        
        for k = 1:nneurons
            current_length = size(training_data(i,j).spikes(k, :), 2);
            if current_length < t_max
                training_data(i,j).spikes(k, (current_length+1):t_max) = 0;
                training_data(i,j).handPos(:,(current_length+1):t_max) = 0; % repmat(training_data(i,j).handPos(:,current_length), 1, (t_max-current_length)); 
            end
        end
    end
end

%% Feature Extraction
% Extracts spike rates and kinematic features (velocity and position) for model training
for dirn = 1:ndire
    % Loop over each direction
    for t = 1:twin
        % Loop over each time bin
        for neuron = 1:nneurons
            % Loop over each neuron
            for n = 1:ntrial
                % Loop over each trial
                t_map = 300 + dt*t;
                % Calculate spike rates for each neuron within each time bin
                total_spikes = sum(training_data(n,dirn).spikes(neuron,t_map:t_map+dt));
                spike_rate = cat(2, spike_rate, total_spikes/(dt*0.001));
                
                % Calculate velocities and positions if first neuron (done once per trial)
                if neuron == 1
                    x1 = training_data(n,dirn).handPos(1,t_map) / 100;
                    x2 = training_data(n,dirn).handPos(1,t_map+dt)/ 100;
                    y1 = training_data(n,dirn).handPos(2,t_map)/ 100;
                    y2 = training_data(n,dirn).handPos(2,t_map+dt)/ 100;
                    
                    vx = (x2 - x1)*0.01 / (dt*0.001);
                    vy = (y2 - y1)*0.01 / (dt*0.001);
                    if (vx+vy) == 0
                        trialid(n) = 1;
                    else
                        % Store displacement and velocities
                        temp_x = cat(2,temp_x, x1);
                        temp_y = cat(2,temp_y, y1);
                        temp_vx = cat(2, temp_vx, vx);
                        temp_vy = cat(2, temp_vy, vy);
                    end
                end
                
            end
            
            % Aggregate firing rates for feature extraction
            firing_rates = cat(1, firing_rates, spike_rate);
            
            % Reset temporary variables for next iteration
            spike_rate = [];

        end
        % Store processed data for each direction and each time bin
        % firing_rates(firing_rates==0) = 1e-4;
        trialid = ~trialid;
        trainingData{dirn}{t} = firing_rates(:,trialid) - mean(firing_rates(:,trialid),"all");
 
        % Calculate and store kinematic data and center positions for each direction and each time bin
        %id = find(temp_vx&temp_vy); % find trials not being padded with 0;
        if length(find(temp_vx+temp_vy)) < 50
            win = min([win,t-1]);
        else
            kinematics{dirn}{t} = [temp_x-mean(temp_x); temp_y-mean(temp_y); temp_vx-mean(temp_vx); temp_vy-mean(temp_vy)]'; 
            centers{dirn}{t} = [mean(temp_x) , mean(temp_y), mean(firing_rates(:,trialid),"all")];

        end
        % Reset firing_rate
        firing_rates = [];
        temp_vx = [];
        temp_vy = [];
        temp_y = [];
        temp_x = [];
        trialid = zeros([1,ntrial]);
    end
    %nonEmptyCells = cellfun(@(c) ~isempty(c), kinematics{dirn});
    time_window(dirn) = win ; %sum(nonEmptyCells);
    win = twin;
end
%% KNN Classifier Training
    % Calculate total number of spikes for each neuron within the first 320ms of each trial
    spikes = [];
    reach_angle = [];
    spike_count = zeros(ntrial, nneurons);
    for dirn = 1:ndire
        for neuron = 1:nneurons
            for n = 1:ntrial
                total_spikes = sum(training_data(n,dirn).spikes(neuron, 1:320));
                spike_count(n, neuron) = total_spikes;
            end
        end
        spikes = cat(1, spikes, spike_count); % Compile spike count data
        reaching_angle = repmat(dirn, ntrial, 1); % Label data with direction
        reach_angle = cat(1, reach_angle, reaching_angle); 
    end

    K = 5; % Number of nearest neighbors for KNN
    customKNNModel = customFitKNN(spikes, reach_angle, K); % Train KNN model

    spikes = [];
    reach_angle = [];    

%% Kalman Filter Training
    kalfilter = kalman_fit(kinematics,trainingData,centers,time_window); % Train Kalman filter using kinematic data
    % Store the trained model in modelParameters
    modelParameters = struct('knnModel', customKNNModel, 'kalModel', kalfilter, 'positionAverage', kinematics, 'timeBin', time_window); % 'linear', beta,
end


function [filter] = kalman_fit(train,response,cent,twindow)
    % KALMAN_FIT Trains Kalman filters for each direction based on the given training data.
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
    filter = {{}};

    for dirn = 1:8
        for twin = 1:(twindow(dirn)-1)
            spike_train = response{dirn}{twin}; % Extract spike train data for the current direction
            kim = train{dirn}{twin}; % Extract kinematic data for the current direction
            kim_next = train{dirn}{twin+1};
            % Compute the variables required for filtering.
            % Preparing data for least squares solution
            endlen = min([length(kim(:,1)), length(kim_next(:,1))]);
            % epsilon = 1; % Small regularization factor
            X2 = kim_next(1:endlen, :)'; % Kinematics at time t
            X1 = kim(1:endlen, :)'; % Kinematics at time t -1

            % Compute the state transition matrix A using least squares
            % This matrix predicts the next state based on the current state
            A = X2*X1'/((X1*X1')); 
            % A = A + epsilon * eye(size(A));
            % Compute the process noise covariance matrix W
            % This matrix represents the uncertainty in the model prediction
            W = (X2 - A*X1)*(X2 - A*X1)'/length(X2(1,:)); 
    
            % The least-squares-optimal transformation from kinematics to neural firing rate
            X = X1; % Kinematic data
            [~,idxz] = sort(sum(spike_train));
            Z = spike_train(:,idxz(1:endlen)); %normalize(spike_train(:,idxz(1:endlen)),'norm',1); % Neural firing data
                

            % Compute the observation matrix H using least squares
            % This matrix relates the state to the observed neural firing rates
            H = Z*X'/(X*X'); 
    
            % Compute the observation noise covariance matrix Q
            % This matrix represents the uncertainty in the measurements (spike counts)
            Q = (Z - H*X)*(Z - H*X)'/length(Z(1,:)); 
            
            % Regularization to avoid singular matrices
            % epsilon = 1; % Small regularization factor
            % epsilon = '1' - assume perfect linearity to minimize the random position error.
    
            % W = W + epsilon * eye(size(W)); % Regularize process noise covariance   inspect 'normalize(W,'center','mean')
            % Q = Q + epsilon * eye(size(Q)); % Regularize measurement noise covariance   inspect 'normalize(Q,'center','mean')
    
            % Store the computed matrices in the filter struct for the current direction
            filter{dirn}{twin}.Q = Q;
            filter{dirn}{twin}.H = H;
            filter{dirn}{twin}.A = A;
            filter{dirn}{twin}.W = W;
            % Store the mean position for the current direction
            filter{dirn}{twin}.center = cent{dirn}{twin};
        end
    end
end


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