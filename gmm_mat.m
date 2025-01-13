close all
clear all
clc


%  load('RealEnv\Direc915Test.mat');  % test_estimat_1
% load('RealEnv\Person915Test.mat'); % test_estimat_2
load('RealEnv_Data\Vehicle915Test.mat'); % test_estimat_3

N = 8000;
FakeCluster = 8;
TrueCluster = 4;
real_Iterations = 300;
Iterations = real_Iterations + 1;
Max_K = 100;
num_components = 4; % 指定 GMM 的组件数量

% real_s = strucDirec.s(1:600);
% real_sigma = strucDirec.sigma(1:600);
% real_k = strucDirec.k(1:600);

% real_s = strucMovingPerson.s(1:600);
% real_sigma = strucMovingPerson.sigma(1:600);
% real_k = strucMovingPerson.k(1:600);

real_s = strucMovingVehicle.s(1:600);
real_sigma = strucMovingVehicle.sigma(1:600);
real_k = strucMovingVehicle.k(1:600);     

for n = 1:600
    % 使用 fitgmdist 函数拟合 GMM
    [data, Data] = RealDataGenerator(N, TrueCluster, pi/8, real_s(n), real_sigma(n));
    data = data*600;

    gmm = fitgmdist(data, num_components);
    u = gmm.mu;
    s = gmm.Sigma;
    estimat_s(n) = mean(sqrt(power(u(1:num_components, 1)./600, 2) + power(u(1:num_components, 2)./600, 2)));
    estimat_sigma(n)=sqrt(mean(s(1,1,1:num_components)./360000 + s(2,2,1:num_components)./360000)/2);
    estimat_k(n) = power(estimat_s(n),2)/(2*power(estimat_sigma(n), 2));

    % 显示拟合后的结果
    disp('拟合后的组件均值：');
    disp(gmm.mu);
    disp('拟合后的组件协方差矩阵：');
    disp(gmm.Sigma);
    disp('拟合后的组件权重：');
    disp(gmm.ComponentProportion);

    figure(1)
    hold off
    scatter(data(:,1),data(:,2));
    hold on
    scatter(u(:,1),u(:,2),200,'MarkerFaceColor','r');
    xlabel('In-phase');
    ylabel('Quadrature');

    %% 绘制参数拟合图
    % s和sigma拟合
    figure(21)
    % 左轴
    yyaxis left
    x_frame = 1:n;
    plot(x_frame, real_s(1:n), '-', 'Color', '#8ECFC9', 'Marker', 'none');  % 绘制真实s
    hold on
    plot(x_frame, estimat_s(1:n), '-', 'Color', '#FFBE7A', 'Marker', 'none');  % 绘制估计s
    ylabel('s');
    % ylim([min_s, max_s]);  % 设置左轴范围
    % yticks(min_s:5e-4:max_s);  % 设置左轴刻度
    % 右轴
    yyaxis right
    plot(x_frame, real_sigma(1:n), '-', 'Color', '#FA7F6F', 'Marker', 'none');  % 绘制真实sigma
    hold on
    plot(x_frame, estimat_sigma(1:n), '-', 'Color', '#82B0D2', 'Marker', 'none');  % 绘制估计sigma
    ylabel('sigma');
    % ylim([min_sigma, max_sigma]);  % 设置右轴范围为真实sigma的最大值
    % yticks(min_sigma:5e-5:max_sigma);  % 设置右轴刻度
    xlabel('frame');
    % 图例
    legend('s-True','s-GMM','sigma-True','sigma-GMM');

    % k拟合
    figure(22)
    plot(x_frame, real_k(1:n), '-', 'Color', '#BEB8DC', 'Marker', 'none');  % 绘制真实k
    hold on
    plot(x_frame, estimat_k(1:n), '-', 'Color', '#E7DAD2', 'Marker', 'none');  % 绘制估计k
    ylabel('k');
    % ylim([min_k, max_k]);
    % yticks(min_k:1:max_k);
    % 图例
    legend('k-True', 'k-GMM');
    xlabel('frame');

    save('gmm_estimat_3.mat') % 保存当前所有变量到文件
end


