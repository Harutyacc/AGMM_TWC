close all
clear all
clc

%% direct
load('gmm_estimat_1.mat')
gmm_k = estimat_k';
gmm_s = estimat_s';
gmm_sigma = estimat_sigma';

load('amm_estimat_1.mat')

% 真实值
actual = real_k(1:600);

% 预测值
predicted_gmm = gmm_k(1:600);
predicted_agmm = em_k(1:600);

% 计算差值
errors_gmm = predicted_gmm - actual;
errors_agmm = predicted_agmm - actual;

% 计算平方差
squared_errors_gmm = errors_gmm .^ 2;
squared_errors_agmm = errors_agmm .^ 2;

% 计算平均平方误差（MSE）
mse_gmm = mean(squared_errors_gmm);
mse_agmm = mean(squared_errors_agmm);

% 计算 RMSE
rmse_gmm_direct = sqrt(mse_gmm);
rmse_agmm_direct = sqrt(mse_agmm);
%% person
load('gmm_estimat_2.mat')
gmm_k = estimat_k';
gmm_s = estimat_s';
gmm_sigma = estimat_sigma';

load('amm_estimat_2.mat')

% 真实值
actual = real_k(1:600);

% 预测值
predicted_gmm = gmm_k(1:600);
predicted_agmm = em_k(1:600);

% 计算差值
errors_gmm = predicted_gmm - actual;
errors_agmm = predicted_agmm - actual;

% 计算平方差
squared_errors_gmm = errors_gmm .^ 2;
squared_errors_agmm = errors_agmm .^ 2;

% 计算平均平方误差（MSE）
mse_gmm = mean(squared_errors_gmm);
mse_agmm = mean(squared_errors_agmm);

% 计算 RMSE
rmse_gmm_person = sqrt(mse_gmm);
rmse_agmm_person = sqrt(mse_agmm);

%% vehicle
load('gmm_estimat_3.mat')
gmm_k = estimat_k';
gmm_s = estimat_s';
gmm_sigma = estimat_sigma';

load('amm_estimat_3.mat')

% 真实值
actual = real_k(1:600);

% 预测值
predicted_gmm = gmm_k(1:600);
predicted_agmm = em_k(1:600);

% 计算差值
errors_gmm = predicted_gmm - actual;
errors_agmm = predicted_agmm - actual;

% 计算平方差
squared_errors_gmm = errors_gmm .^ 2;
squared_errors_agmm = errors_agmm .^ 2;

% 计算平均平方误差（MSE）
mse_gmm = mean(squared_errors_gmm);
mse_agmm = mean(squared_errors_agmm);

% 计算 RMSE
rmse_gmm_vehicle = sqrt(mse_gmm);
rmse_agmm_vehicle = sqrt(mse_agmm);


