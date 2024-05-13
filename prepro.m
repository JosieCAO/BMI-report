% Get the number of neurons and time steps
[num_neurons, num_bins] = size(trial(1, 1).spikes);
% Create a new figure window
figure;

% Define the color used for plotting
color = [0, 0, 0]; % Black

% Loop through each neuron
for neuron = 1:num_neurons
    % Find the spike data for the current neuron
    spikes = trial(1, 1).spikes(neuron, :);
    
    % Get the spike times for the current neuron
    spike_times = find(spikes);
    
    % Plot the spikes for the current neuron in the raster plot
    scatter(spike_times, ones(size(spike_times)) * neuron, 3, color, 'filled');
    hold on;
end

% Draw red dashed lines at 300 ms and 572 ms
line([300 300], ylim, 'Color', 'red', 'LineWidth', 1, 'LineStyle', '--');
line([572 572], ylim, 'Color', 'red', 'LineWidth', 1, 'LineStyle', '--');

% Label "Moving Phase"
text(425, num_neurons + 9, 'Moving Phase', 'Color', 'red', 'FontSize', 10, 'HorizontalAlignment', 'center');

% Set graph properties
xlabel('Time (ms)');
ylabel('Neurons');
% title('Population Raster Plot for a Single Trial');
xlim([0, 700]); % Set the x-axis range from 0 to 700
ylim([0, 100]); % Set the y-axis range from 0 to 100
set(gca, 'PlotBoxAspectRatio', [3 1 1]); % Adjust the aspect ratio

% Keep all added graphical elements
hold off;


% There are 8 directions and selected neuron IDs
directions = 1:8; 
neuron_ids = [65, 73, 75]; % Select the neuron IDs for which to plot tuning curves
colors = lines(length(neuron_ids)); % Generate different colors to distinguish different neurons

figure; % Create a new figure window
hold on; % Hold the figure, to plot all neurons' tuning curves in the same graph

% Loop through the selected neurons
for i = 1:length(neuron_ids)
    neuron_id = neuron_ids(i);
    firing_rates = zeros(1, length(directions)); % Initialize an array to store firing rates
    
    % Loop through all directions
    for d = 1:length(directions)
        dir = directions(d);
        total_spikes = 0;
        num_trials = 0;
        
        % Loop through all trials for this direction
        for trial_idx = 1:length(trial)
            if ~isempty(trial(trial_idx, dir).spikes)
                spikes = trial(trial_idx, dir).spikes(neuron_id, :);
                firing_rate = sum(spikes) / (length(spikes) / 1000); % Calculate firing rate in spikes per second
                total_spikes = total_spikes + firing_rate;
                num_trials = num_trials + 1;
            end
        end
        
        % Calculate average firing rate
        firing_rates(d) = total_spikes / num_trials;
    end
    
    % Plot the tuning curve for the current neuron, using solid points and a smooth line
    plot(directions, firing_rates, 'Color', colors(i,:), 'LineWidth', 2, 'Marker', 'o', 'MarkerFaceColor', colors(i,:), 'LineStyle', '-');
end

% Set graph properties
xlabel('Direction');
ylabel('Firing Rate (spikes/s)');
title('Tuning Curves of Neurons');
set(gca, 'PlotBoxAspectRatio', [3 1 1]); 
legend(arrayfun(@(id) sprintf('Neuron %d', id), neuron_ids, 'UniformOutput', false));
hold off;