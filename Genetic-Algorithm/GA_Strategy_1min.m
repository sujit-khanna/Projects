
%% Load NIFTY data

[a b]=xlsread('HFT data1.xlsx','Min');
data=a;

testPts = floor(0.9*length(data(:,4)));
step =5; % 30 minute interval
Nifty = data(1:step:testPts,2:end);
NiftyV = data(testPts+1:step:end,2:end);
%annualScaling = sqrt(250);
annualScaling = sqrt(250*60*11/step);
%cost = 0.01;
cost = 0.0;

%% Williams %R
w = willpctr(Nifty,14);
plot(w)

%% Williams %R trading strategy
wpr(Nifty,100,annualScaling,cost)

%% WPR performance
range = {1:500};
wfun = @(x) wprFun(x,Nifty,annualScaling,cost);
tic
[maxSharpe,param,sh] = parameterSweep(wfun,range);
toc
wpr(Nifty,param,annualScaling,cost)
%figure
%plot(sh)
%ylabel('Sharpe''s Ratio')

%% Generate trading signals
%N = 12; M = 26; thresh = 55; P = 2; Q = 104; R=3;
N = 12; M = 26; thresh = 55; P = 2; Q = 55; R=2;
sma = leadlag(Nifty(:,end),N,M,annualScaling,cost);
srs = rsi(Nifty(:,end),annualScaling,cost);
%srs = rsi(Bund(:,end),[Q R],annualScaling,cost);
swr = wpr(Nifty,param,annualScaling,cost);

signals = [sma srs swr];
names = {'MA','RSI','WPR'};
%% Trading signals
% Plot the "state" of the market represented by the signals
figure
ax(1) = subplot(2,1,1); plot(Nifty(:,end));
ax(2) = subplot(2,1,2); imagesc(signals')
cmap = colormap([1 0 0; 0 0 1; 0 1 0]);
set(gca,'YTick',1:length(names),'YTickLabel',names);
linkaxes(ax,'x');

%% Generate initial population
% Generate initial population for signals
close all
I = size(signals,2);
pop = initializePopulation(I);
imagesc(pop)
xlabel('Bit Position'); ylabel('Individual in Population')
colormap([1 0 0; 0 1 0]); set(gca,'XTick',1:size(pop,2))
%% Fitness Function
% Objective is to find a target bitstring (minimum value)
type fitness
%%
% Objective function definition
obj = @(pop) fitness(pop,signals,Nifty(:,end),annualScaling,cost)
%%
% Evalute objective for population
obj(pop)
%% Solve With Genetic Algorithm
% Find best trading rule and maximum Sharpe ratio (min -Sharpe ratio)
options = gaoptimset('Display','iter','PopulationType','bitstring',...
    'PopulationSize',size(pop,1),...
    'InitialPopulation',pop,...
    'CrossoverFcn', @crossover,...
    'MutationFcn', @mutation,...
    'PlotFcns', @plotRules,...
    'Vectorized','on');

[best,minSh] = ga(obj,size(pop,2),[],[],[],[],[],[],[],options)

%% Evaluate Best Performer
%%In sample perfromance
s = tradeSignal(best,signals);
s = (s*2-1); % scale to +/-1
r  = [0; s(1:end-1).*diff(Nifty(:,end))-abs(diff(s))*cost/2];
sh = annualScaling*sharpe(r,0);
r1= s(1:end-1);
r2=diff(Nifty(:,end));
r3=abs(diff(s));
r4=cost/2;

% Plot results
figure
ax(1) = subplot(2,1,1);
plot(Nifty(:,end))
title(['Evolutionary Learning Resutls, Sharpe Ratio = ',num2str(sh,3)])
ax(2) = subplot(2,1,2);
plot([s,cumsum(r)])
legend('Position','Cumulative Return')
title(['Final Return = ',num2str(sum(r),3), ...
    ' (',num2str(sum(r)/Nifty(1,end)*100,3),'%)'])
linkaxes(ax,'x');

z1=num2str(sum(r)/NiftyV(1,end)*100,3);
%xlswrite('output.xlsx',z1,'Sheet1');

%%Out-of-Sample performance
sma = leadlag(NiftyV(:,end),N,M,annualScaling,cost);
srs = rsi(NiftyV(:,end),[P Q],thresh,annualScaling,cost);
swr = wpr(NiftyV,param,annualScaling,cost);
signals = [sma srs swr];

s = tradeSignal(best,signals);
s = (s*2-1); % scale to +/-1
r  = [0; s(1:end-1).*diff(NiftyV(:,end))-abs(diff(s))*cost/2];
sh = annualScaling*sharpe(r,0);
r1=s(1:end-1);
r2=diff(NiftyV(:,end));
r3=abs(diff(s));
r4=cost/2;

% Plot results
figure
ax(1) = subplot(2,1,1);
plot(NiftyV(:,end))
title(['Evolutionary Learning Resutls, Sharpe Ratio = ',num2str(sh,3)])
ax(2) = subplot(2,1,2);
plot([s,cumsum(r)])
legend('Position','Cumulative Return')
title(['Final Return = ',num2str(sum(r),3), ...
    ' (',num2str(sum(r)/NiftyV(1,end)*100,3),'%)'])
linkaxes(ax,'x');

z2=num2str(sum(r)/NiftyV(1,end)*100,3);
%xlswrite('output.xlsx',z2,'Sheet2');
