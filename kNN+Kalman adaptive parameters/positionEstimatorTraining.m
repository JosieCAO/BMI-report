%%% Team NAME : bls
%%% Team Members: Josephine Cao, Jiayu Liu, Xinyi Liu, Fangyuan Wang
%%% BMI Spring 2024 (Update 17th March 2024)

function [modelParameters] = positionEstimatorTraining(trainingdata)
% POSITIONESTIMATORTRAINING Trains the model parameters for estimating hand position.
% Input:
%   training_data - Struct containing the neural and hand position data for training.
% Output:
%   modelParameters - Struct containing the trained model parameters.

%% Initialize variables for data processing and feature extraction
ntrial = size(trainingdata, 1);
ndire = size(trainingdata, 2);
nneurons = size(trainingdata(1,1).spikes, 1); % Fixed neuron size
binsize = 20; % Time bin size for spike rate calculation
store_sr = []; % Temporary storage for spike rates
spikerates = []; % Collection of firing rates across all trials and neurons
store_vx = []; % Temporary storage for velocity in x-direction
store_vy = []; % Temporary storage for velocity in y-direction
disp_x = []; % Displacement in x-direction
disp_y = []; % Displacement in y-direction
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
        leng = size(trainingdata(i,j).spikes(1, :),2);
        current_length = [current_length, leng];
    end
end
t_max = floor(max(current_length)/binsize)*binsize;
twin = ((t_max-binsize-300)/binsize);
win = twin;
% Pad the data with zeros to make all trials of minimun equal length
for j = 1:ndire
    for i = 1:ntrial
        
        for k = 1:nneurons
            current_length = size(trainingdata(i,j).spikes(k, :), 2);
            if current_length < t_max
                trainingdata(i,j).spikes(k, (current_length+1):t_max) = 0;
                trainingdata(i,j).handPos(:,(current_length+1):t_max) = 0; % repmat(training_data(i,j).handPos(:,current_length), 1, (t_max-current_length)); 
            end
        end
    end
end

%% Feature Extraction
% Extracts spike rates and kinematic features (velocity and position) for model training
for angle = 1:ndire
    % Loop over each direction
    for t = 1:twin
        % Loop over each time bin
        for neuron = 1:nneurons
            % Loop over each neuron
            for n = 1:ntrial
                % Loop over each trial
                t_map = 300 + binsize*t;
                % Calculate spike rates for each neuron within each time bin
                total_spikes = sum(trainingdata(n,angle).spikes(neuron,t_map:t_map+binsize));
                store_sr = cat(2, store_sr, total_spikes/(binsize*0.001));
                
                % Calculate velocities and positions if first neuron (done once per trial)
                if neuron == 1
                    x1 = trainingdata(n,angle).handPos(1,t_map) / 100;
                    x2 = trainingdata(n,angle).handPos(1,t_map+binsize)/ 100;
                    y1 = trainingdata(n,angle).handPos(2,t_map)/ 100;
                    y2 = trainingdata(n,angle).handPos(2,t_map+binsize)/ 100;
                    
                    velocityX = (x2 - x1)*0.01 / (binsize*0.001);
                    velocityY = (y2 - y1)*0.01 / (binsize*0.001);
                    if (velocityX+velocityY) == 0
                        trialid(n) = 1;
                    else
                        % Store displacement and velocities
                        disp_x = cat(2,disp_x, x1);
                        disp_y = cat(2,disp_y, y1);
                        store_vx = cat(2, store_vx, velocityX);
                        store_vy = cat(2, store_vy, velocityY);
                    end
                end
                
            end
            
            % Aggregate firing rates for feature extraction
            spikerates = cat(1, spikerates, store_sr);
            
            % Reset temporary variables for next iteration
            store_sr = [];

        end
        % Store processed data for each direction and each time bin
        % firing_rates(firing_rates==0) = 1e-4;
        trialid = ~trialid;
        trainingData{angle}{t} = spikerates(:,trialid) - mean(spikerates(:,trialid),"all");
 
        % Calculate and store kinematic data and center positions for each direction and each time bin
        %id = find(temp_vx&temp_vy); % find trials not being padded with 0;
        if length(find(store_vx+store_vy)) < 50
            win = min([win,t-1]);
        else
            kinematics{angle}{t} = [disp_x-mean(disp_x); disp_y-mean(disp_y); store_vx-mean(store_vx); store_vy-mean(store_vy)]'; 
            centers{angle}{t} = [mean(disp_x) , mean(disp_y), mean(spikerates(:,trialid),"all")];

        end
        % Reset firing_rate
        spikerates = [];
        store_vx = [];
        store_vy = [];
        disp_y = [];
        disp_x = [];
        trialid = zeros([1,ntrial]);
    end
    %nonEmptyCells = cellfun(@(c) ~isempty(c), kinematics{dirn});
    time_window(angle) = win ; %sum(nonEmptyCells);
    win = twin;
end
%% KNN Classifier Training
    % Calculate total number of spikes for each neuron within the first 320ms of each trial
    features = [];
    labels = [];
    spikeCounts = zeros(ntrial, nneurons);
    for angle = 1:ndire
        for neuron = 1:nneurons
            for n = 1:ntrial
                total_spikes = sum(trainingdata(n,angle).spikes(neuron, 1:320));
                spikeCounts(n, neuron) = total_spikes;
            end
        end
        features = cat(1, features, spikeCounts); % Compile spike count data
        Lable_angles = repmat(angle, ntrial, 1); % Label data with direction
        labels = cat(1, labels, Lable_angles); 
    end

    K = 10; % Number of nearest neighbors for KNN
    customKNNModel = customFitKNN(features, labels, K); % Train KNN model

    features = [];
    labels = [];    

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

    for angle = 1:8
        for twin = 1:(twindow(angle)-1)
            spike_train = response{angle}{twin}; % Extract spike train data for the current direction
            kim = train{angle}{twin}; % Extract kinematic data for the current direction
            kim_next = train{angle}{twin+1};
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
            filter{angle}{twin}.Q = Q;
            filter{angle}{twin}.H = H;
            filter{angle}{twin}.A = A;
            filter{angle}{twin}.W = W;
            % Store the mean position for the current direction
            filter{angle}{twin}.center = cent{angle}{twin};
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