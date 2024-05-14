% Test Script to give to the students, March 2015
%% Continuous Position Estimator Test Script
% This function first calls the function "positionEstimatorTraining" to get
% the relevant modelParameters, and then calls the function
% "positionEstimator" to decode the trajectory. 

function RMSE = testFunction_for_students_MTb(teamName)

load monkeydata_training.mat

% Set random number generator
rng(2013);
ix = randperm(length(trial));

%addpath(teamName);

% Select training and testing data (you can choose to split your data in a different way if you wish)
trainingData = trial(ix(1:50),:);
testData = trial(ix(51:end),:);

fprintf('Testing the continuous position estimator...')

meanSqError = 0;
n_predictions = 0;  
real = [];
supreal = [];
supos = [];
supdir = [];
supd = [];

figure(2)
hold on
axis square
grid

% Train Model
modelParameters = positionEstimatorTraining(trainingData);
num = 0;
num_fix_dir = 0;
for tr=1:size(testData,1)
    display(['Decoding block ',num2str(tr),' out of ',num2str(size(testData,1))]);
    pause(0.001)
    
    for direc=randperm(8) 
        decodedHandPos = [];
        real = [];
        times=320:20:size(testData(tr,direc).spikes,2);
        num = num+ 1;
        for t=times
            past_current_trial.trialId = testData(tr,direc).trialId;
            past_current_trial.spikes = testData(tr,direc).spikes(:,1:t); 
            past_current_trial.decodedHandPos = decodedHandPos;

            past_current_trial.startHandPos = testData(tr,direc).handPos(1:2,1); 
            
            if nargout('positionEstimator') == 3
                [decodedPosX, decodedPosY, newParameters] = positionEstimator(past_current_trial, modelParameters);
                modelParameters = newParameters;
            elseif nargout('positionEstimator') == 2
                [decodedPosX, decodedPosY] = positionEstimator(past_current_trial, modelParameters);
            end
            
            decodedPos = [decodedPosX; decodedPosY];
            decodedHandPos = [decodedHandPos decodedPos];
            real = [real, testData(tr,direc).handPos(1:2,t)];

            meanSqError = meanSqError + norm(testData(tr,direc).handPos(1:2,t) - decodedPos)^2;
            
        end
        n_predictions = n_predictions+length(times);
        hold on
        plot(decodedHandPos(1,:),decodedHandPos(2,:), 'r');
        plot(testData(tr,direc).handPos(1,times),testData(tr,direc).handPos(2,times),'b')
        if mod(num,5)==0
            supreal = [supreal real];
            supos = [supos decodedHandPos];
        end
        % to plot xy position of direction 4
        if direc == 4
            supdir = [supdir real];
            supd = [supd decodedHandPos];
            num_fix_dir = num_fix_dir + 1;
        end
    end
end
hold off
legend('Decoded Position', 'Actual Position')


figure(3)
t = 1:length(supreal(1,:));
plot3(t,supos(1,:),supos(2,:),'b')
hold on 
plot3(t,supreal(1,:),supreal(2,:),'r')
hold off
xlabel('time(s)')
ylabel('x position(cm)')
zlabel('y position(cm)')
title('sampled hand position vs time')
legend('Decoded Position', 'Actual Position')

RMSE = sqrt(meanSqError/n_predictions);

figure(4)
t = 1:length(supdir(1,:));
plot3(t,supd(1,:),supd(2,:),'b')
hold on 
plot3(t,supdir(1,:),supdir(2,:),'r')
hold off
xlabel('time(s)')
ylabel('x position(cm)')
zlabel('y position(cm)')
title('sampled hand position of direction 4 vs time')
legend('Decoded Position', 'Actual Position')
%rmpath(genpath(teamName))

% num_fix_dir

end

