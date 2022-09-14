clear all; close all; clc;
%% Define data points for camera 0, camera 1, and radar
% Camera 0 coordinates (pic 8): point01. The first 0 is for cam0 then 
% the number of the sample. Example calib 1 from camera0 is point01 etc.
point01 = [381 405 1]; 
point02 = [295 383 1];
point03 = [396 331 1];
point04 = [574 437 1];
point05 = [473 349 1];
point06 = [351 352 1];
point07 = [286 445 1];
point08 = [498 440 1];
point09 = [218 431 1];
point010 = [240 388 1];
point011 = [489 384 1];
point012 = [385 368 1];
point013 = [269 352 1];
point014 = [434 344 1];
point015 = [354 330 1];
point016 = [398 310 1];
point017 = [310 302 1];
point018 = [417 289 1];
point019 = [356 281 1];

cam0 = [point01; point02; point03; point04; point05; point06; point07; 
        point08; point09; point010; point011; point012; point013; point014; 
        point015; point016; point017; point018; point019];

% Camera 1 coordinates (pic 6): point11. The first "1" is for cam1 then 
% the number of the sample. Example calib 1 from camera1 is point11 etc.
point11 = [245 382 1];
point12 = [173 357 1];
point13 = [303 314 1];
point14 = [429 436 1];
point15 = [368 337 1];
point16 = [243 331 1];
point17 = [131 409 1];
point18 = [352 430 1];
point19 = [91 392 1];
point110 =[129 356 1];
point111 =[373 371 1];
point112 =[276 348 1];
point113 =[174 326 1];
point114 =[337 328 1];
point115 =[266 308 1];
point116 =[321 292 1];
point117 =[238 281 1];
point118 =[350 272 1];
point119 =[295 263 1];

cam1 = [point11; point12; point13; point14; point15; point16; point17; 
        point18; point19; point110; point111; point112; point113; point114; 
        point115; point116; point117; point118; point119];


% Radar Data (measured ground truth in inches)
point1 = [77.5 0 1];
point2 = [80.5 -15 1];
point3 = [130 9 1];
point4 = [60 29 1];
point5 = [111 24 1];
point6 = [110.5 -4.5 1];
point7 = [63 -15.5 1];
point8 = [68.5 16.5 1];
point9 = [72 -28.5 1];
point10 = [92 -29 1];
point11 = [95 23.5 1];
point12 = [107 3 1];
point13 = [118 -25 1];
point14 = [125.5 17 1];
point15 = [143.5 -4 1];
point16 = [166.5 11 1];
point17 = [181 -19 1];
point18 = [205 25 1];
point19 = [225 0 1];

% % Radar Data (peak detect from radar speactrum)
% point1 = [2.134977 0.000000 1];
% point2 = [2.399066 -0.395826 1];
% point3 = [3.431303 0.239978 1];
% point4 = [1.669624 0.769585 1];
% point5 = [2.971744 0.562707 1];
% point6 = [2.962036 -0.137918 1];
% point7 = [1.730367 -0.413755 1];
% point8 = [1.891943 0.500644 1];
% point9 = [2.018650 -0.695109 1];
% point10 = [2.505746 -0.728209 1];
% point11 = [2.537872 0.606841 1];
% point12 = [2.898859 0.202740 1];
% point13 = [3.204822 -0.606841 1];
% point14 = [3.298546 0.386171 1];
% point15 = [3.794487 -0.088268 1];
% point16 = [4.345630 0.612358 1];
% point17 = [4.739258 -0.220669 1];
% point18 = [5.302524 1.004046 1];
% point19 = [5.864831 0.273078 1];


radar = [point1; point2; point3; point4; point5; point6; point7; point8; 
        point9; point10; point11; point12; point13; point14; point15; point16; 
        point17; point18; point19];

N = 19; % Number of data points
test_results = [];
diff_results = [];

%% Finding H
for i=1:N
    if i > 1 && i < N
        train_cam0 = [cam0(1:i-1,:); cam0(i+1:end,:)];
        train_radar = [radar(1:i-1,:); radar(i+1:end,:)];
    elseif i == 1
        train_cam0 = cam0(i+1:end,:);
        train_radar = radar(i+1:end,:);
    elseif i == N
        train_cam0 = cam0(1:i-1,:);
        train_radar = radar(1:i-1,:);
    end
    test_cam0 = cam0(i:i,:);
    test_radar = radar(i:i,:);
    
    %Find H from cam0 to radar
    HRadar0 = findH(train_cam0, train_radar);

    %% Calculate transformations

    % Transform cam0 data to radar coordinates
    calculatedRadar0 = transform(HRadar0, test_cam0);

    %% Test Results

    % This is calculating the percent difference between the calculated Camera0
    % data and the original cam0 data.
    pDiff0 = evaluation(calculatedRadar0, test_radar',1);

    % This is calculating the difference between the calculated Camera1 data
    % and the original cam1 data.
    diff1 = evaluation(calculatedRadar0, test_radar', 2);
    test_results = [test_results, calculatedRadar0];
    diff_results = [diff_results, diff1];
end

%% Plot figure for paper
grid on
hold on
set(gca,'FontSize',15)
xlabel('x-axis (inches)','FontSize', 15)
ylabel('y-axis (inches)','FontSize', 15)
xlim([40 240]);
ylim([-30 40]);
scatter(radar(:,1), radar(:,2), 100, 'blue','filled');
scatter(test_results(1,:), test_results(2,:), 100, 'red', 'filled');
legend('ground truth','predicted radar coordinate');

diffs = reshape(diff_results(1:2, :),1,[]);
mean(diffs)
max(diffs)

%% Final H using all data, just for testing the real data

HRadar0_final = findH(cam0, radar)
save('cam0_to_radar.mat', 'HRadar0_final')
