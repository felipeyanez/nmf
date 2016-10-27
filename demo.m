%% Sierra Project-Team
%  Author: Felipe Yanez
%  Copyright (c) 2014-2016

close all;
clear all;
clc;

rand('seed',0);
randn('seed',0);

path(path,'./Other NMF algorithms');

%% Generate data

% Problem size
n = 200;
m = 500;
r = 10;

% True matrices
W  = abs(randn(n,r));
H  = abs(randn(r,m));
V  = W * H;

% Initialization
W0 = rand(n,r);
H0 = rand(r,m);

% Number of iterations
N = 1e3;

disp('Non-negative matrix factorization for synthetic data');
fprintf('\nProblem size: n = %i; m = %i; r = %i;\n',n,m,r);
fprintf('Number of iterations: %i;\n',N);

%% Multiplicative updates algorithm (MUA)

fprintf('\nMUA stats:\n');

[W1 H1 obj1 time1] = nmf_kl_mua(V,W0,H0,N);

disp(['Objective: ',num2str(obj1(end))]);
disp(['Run time: ',num2str(time1(end)),' s']);

%% Alternating direction method of multipliers (ADMM)

rho = 5;
fprintf('\nADMM (rho = %i) stats:\n',rho);

[W2, H2, obj2, time2] = nmf_kl_admm(V, W0, H0, rho, N);

disp(['Objective: ',num2str(obj2(end))]);
disp(['Run time: ',num2str(time2(end)),' s']);

%% First-order primal-dual algorithm (FPA)

D = 5;
fprintf('\nFPA (D = %i) stats:\n',D);

[W3, H3, obj3, time3] = nmf_kl_fpa(V, W0, H0, N, D);

disp(['Objective: ',num2str(obj3(end))]);
disp(['Run time: ',num2str(time3(end)),' s']);

%% Objective vs iterations

% Colors
c1 = [0.8 0.0 0.0]; % MUA
c2 = [0.0 0.6 0.0]; % ADMM
c3 = [0.0 0.4 0.8]; % FPA primal

str1 = 'MUA';
str2 = ['ADMM (\rho = ',num2str(rho,2),')'];
str3 = ['FPA (D = ',num2str(D),')'];

fig1c = figure;
axes1 = axes('Parent',fig1c,'YScale','log','XScale','log','FontSize',12);
xlim([1 N]); ylim([1e-4 1e8]);
box(axes1,'on');
grid(axes1,'on');
hold(axes1,'all');

loglog1 = loglog(2*N,0, 2*N,0, 2*N,0, 'Parent',axes1);
set(loglog1(1),'MarkerFaceColor', c1, ...
               'MarkerEdgeColor', c1, ...
               'Marker',    'square', ...
               'Color',           c1, ...
               'LineWidth',        1, ...
               'DisplayName', str1);
set(loglog1(2),'MarkerFaceColor', c2, ...
               'MarkerEdgeColor', c2, ...
               'Marker', 'o',         ...
               'Color',           c2, ...
               'LineWidth', 1,        ...
               'DisplayName', str2);
set(loglog1(3),'MarkerFaceColor', c3, ...
               'MarkerEdgeColor', c3, ...
               'Marker', '^',         ...
               'Color',           c3, ...
               'LineWidth', 1,        ...
               'DisplayName', str3);
legend1 = legend(axes1,'show');
set(legend1,'FontSize',12);

ms      = 6;
marks   = 10.^(0:log10(N));
loglog2 = loglog(1:N,obj1, marks,obj1(marks), ...
                 1:N,obj2, marks,obj2(marks), ...
                 1:N,obj3, marks,obj3(marks), 'Parent',axes1);
set(loglog2(1),'Color', c1, 'LineWidth', 1);
set(loglog2(2),'MarkerFaceColor', c1, ...
               'MarkerEdgeColor', c1, ...
               'MarkerSize',      ms, ...
               'Marker',    'square', ...
               'LineStyle','none');
set(loglog2(3),'Color', c2, 'LineWidth', 1);
set(loglog2(4),'MarkerFaceColor', c2, ...
               'MarkerEdgeColor', c2, ...
               'MarkerSize',      ms, ...
               'Marker',         'o', ...
               'LineStyle','none');
set(loglog2(5),'Color', c3, 'LineWidth', 1);
set(loglog2(6),'MarkerFaceColor', c3, ...
               'MarkerEdgeColor', c3, ...
               'MarkerSize',      ms, ...
               'Marker',         '^', ...
               'LineStyle','none');

xlabel('Number of iterations','FontSize',18);
ylabel('Objective function','FontSize',18);