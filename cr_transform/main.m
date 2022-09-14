clear all; close all; clc;
%% Define data points for camera 0, camera 1, and radar
% Camera 0 coordinates (pic 8): point01. The first 0 is for cam0 then 
% the number of the sample. Example calib 1 from camera0 is point01 etc.

cam0 = table2array(readtable('cam_person.csv'));

radar = table2array(readtable('radar_person.csv'));

N = 25; % Number of data points



%% Finding H

% Find H from radar to cam0
% H0 = findH(radar,cam0);
%Find H from cam0 to radar
HRadar0 = findH(cam0, radar);



%load('person_matrix.mat')




%% Calculate transformations


% Transform cam0 data to radar coordinates
calculatedRadar0 = transform(HRadar0, cam0);

%% Test Results
% 
% % This is calculating the percent difference between the calculated Camera0
% % data and the original cam0 data.
% pDiff0 = evaluation(calculatedCAM0,cam0',1);
% 
% % This is calculating the difference between the calculated Camera1 data
% % and the original cam1 data.
% % diff1 = evaluation(calculatedCAM1,cam1',2);
save("cam0_to_radar.mat", "HRadar0");

%% Plot figure for paper

%testingSet0 = [cam0(2:4,:); cam0(end-6:end-1,:)];
%testingSet1 = [cam1(2:4,:); cam1(15:end-1,:)];
testingSet0 = cam0;
%testingSet0 = table2array(readtable('cam_cyclist.csv'));

%radarTest = [radar(2:4,:); radar(end-6:end-1,:)];
radarTest = radar;
%radarTest = table2array(readtable('radar_cyclist.csv'));
calculatedCAM0 = transform(HRadar0, testingSet0);
% calculatedCAM1 = transform(HRadar1, testingSet1);

%disp(calculatedCAM0);
%disp(calculatedCAM0(:,66));
%disp(cam0(66,:));
disp(calculatedCAM0.');
pause;

grid on
hold on
set(gca,'FontSize',18)
xlabel('x inches','FontSize', 18)
ylabel('y inches','FontSize', 18)
%xlim([0 100]);
%ylim([0 100]);
scatter(radarTest(:,1),radarTest(:,2),100,'blue');
scatter(calculatedCAM0(1,:),calculatedCAM0(2,:),100,'red');
% scatter(calculatedCAM1(1,:),calculatedCAM1(2,:),100,'black','filled');
legend('Truth','radar (from cam0)');
title('Transformed radar Coordiantes');
hold off



