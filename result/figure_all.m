close all
clear all
clc

load('gmm_estimat_3.mat')
gmm_k = estimat_k';
gmm_s = estimat_s';
gmm_sigma = estimat_sigma';

load('amm_estimat_3.mat')

%% 绘制参数拟合图
% s和sigma拟合
x_frame = 1:600;

figure(21)
plot(x_frame, real_s(1:600), '-', 'Color', '#FB9968', 'Marker', 'none', 'LineWidth', 0.8);  % 绘制真实s  #FB9968
hold on
plot(x_frame, gmm_s(1:600), '--', 'Color', '#96CCCB', 'Marker', 'none', 'LineWidth', 0.5);
hold on
plot(x_frame, em_s(1:600), ':', 'Color', '#757CBB', 'Marker', 'none', 'LineWidth', 1);  % 绘制估计s#A1A9D0
grid on
ylabel('s');
xlabel('Frame');
set(gca, 'FontSize', 18); 
% 图例
legend('Ground Truth','GMM','AGMM', 'FontSize', 14);
print('vehicle_s.eps', '-depsc', '-r600'); % 以 300 dpi 分辨率保存图形
% ylim([min_s, max_s]);  % 设置左轴范围
% yticks(min_s:5e-4:max_s);  % 设置左轴刻度
% 右轴
figure(22)
plot(x_frame, real_sigma(1:600), '-', 'Color', '#FB9968', 'Marker', 'none', 'LineWidth', 0.8);  % 绘制真实sigma
hold on
plot(x_frame, gmm_sigma(1:600), '--', 'Color', '#96CCCB', 'Marker', 'none', 'LineWidth', 0.5);
hold on
plot(x_frame, em_sigma(1:600), ':', 'Color', '#757CBB', 'Marker', 'none', 'LineWidth', 1);  % 绘制估计sigma
grid on
ylabel('σ');
% ylim([min_sigma, max_sigma]);  % 设置右轴范围为真实sigma的最大值
% yticks(min_sigma:5e-5:max_sigma);  % 设置右轴刻度
xlabel('Frame');
set(gca, 'FontSize', 18); 
% 图例
legend('Ground Truth','GMM','AGMM', 'FontSize', 14);
print('vehicle_sigma.eps', '-depsc', '-r600'); % 以 300 dpi 分辨率保存图形

% k拟合
figure(23)
plot(x_frame, real_k(1:600), '-', 'Color', '#FB9968', 'Marker', 'none', 'LineWidth', 0.8);  % 绘制真实k
hold on
plot(x_frame, gmm_k(1:600), '--', 'Color', '#96CCCB', 'Marker', 'none', 'LineWidth', 0.5)
hold on
plot(x_frame, em_k(1:600), ':', 'Color', '#757CBB', 'Marker', 'none', 'LineWidth', 1);  % 绘制估计k
grid on
ylabel('k');
% ylim([min_k, max_k]);
% yticks(min_k:1:max_k);
% 图例
set(gca, 'FontSize', 18); 
% 图例
legend('Ground Truth','GMM','AGMM', 'FontSize', 14);
xlabel('Frame');
print('vehicle_k.eps', '-depsc', '-r600'); % 以 300 dpi 分辨率保存图形