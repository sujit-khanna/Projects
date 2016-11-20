clear;

[a b]=xlsread('final_data1.xlsx','sheet12');
idxA=a(:,1);  %independent
idxC=a(:,2);  %dependent
x=idxA;
y=idxC;

% Augment x with ones to  accomodate possible offset in the regression
% between y vs x.

x=[x ones(size(x))];

delta=0.0001; % delta=1 gives fastest change in beta, delta=0.000....1 allows no change (like traditional linear regression).

yhat=NaN(size(y)); % measurement prediction
e=NaN(size(y)); % measurement prediction error
eavg=NaN(size(y));
eavg1=NaN(size(y)-50);
e1=NaN(size(y)-50);
Q1=NaN(size(y)-50);
Q=NaN(size(y)); % measurement prediction error variance
eavg=NaN(size(y));% mean measurement prediction error

% initialize R, P and beta.
R=zeros(2);
P=zeros(2);
beta=NaN(2, size(x, 1));
Vw=delta/(1-delta)*eye(2);
Ve=0.001;


% Initialize beta(:, 1) to zero
beta(1, 1)=0.302;
beta(2,1)=0;
%beta(:, 1)=1.73;
% Given initial beta and R (and P)
for t=2:length(y)
    if (t > 1)
        beta(:, t)=beta(:, t-1); % state prediction
        R=P+Vw; % state covariance prediction
    end
    
    yhat(t)=x(t, :)*beta(:, t); % measurement prediction

    Q(t)=x(t, :)*R*x(t, :)'+Ve; % measurement variance prediction
    
    
    % Observe y(t)
    e(t)=y(t)-yhat(t); % measurement prediction error
    
    eavg(t)=nanmean(e(1:t)); 
    K=R*x(t, :)'/Q(t); % Kalman gain
    
    beta(:, t)=beta(:, t)+K*e(t); % State update
    P=R-K*x(t, :)*R; % State covariance update
    
end

%for i=50:length(y)
%eavg(i)=nanmean(e(50:i));
%end
eavg1=eavg(50:end);
e1=e(50:end);
Q1=Q(50:end);
beta1=beta(:,50:end);

%eavg(1)=e(1);
plot(beta(1, :)');

figure;

plot(beta(2, :)');

figure;

plot(e1, 'r');

hold on;
plot(sqrt(Q1));

hold on;
plot(-sqrt(Q1));

hold on;
plot(eavg1,'g');

y2=[x(:, 1) y];
y3=y2(50:end,:);

u=var(beta1(1,:));
v=var(e1);
x=u/v; %signal-to-Noise
z=beta1(1,:)'./e1;

o=1;

%if( abs(z) < sqrt(x))
longsEntry=e1 < -sqrt(Q1); 
longsExit=e1 > eavg1;
%end

shortsEntry=e1 > sqrt(Q1);
shortsExit=e1 < eavg1;

numUnitsLong=NaN(length(y2)-49, 1);
numUnitsShort=NaN(length(y2)-50, 1);


numUnitsLong(1)=0;
numUnitsLong(longsEntry)=1; 
numUnitsLong(longsExit)=0;
numUnitsLong=fillMissingData(numUnitsLong); 

numUnitsShort(1)=0;
numUnitsShort(shortsEntry)=-1; 
numUnitsShort(shortsExit)=0;
numUnitsShort=fillMissingData(numUnitsShort);

numUnits=numUnitsLong+numUnitsShort;
positions=repmat(numUnits, [1 size(y3, 2)]).*[-beta1(1, :)' ones(size(beta1(1, :)'))].*y3; 
pnl=sum(lag(positions, 1).*(y3-lag(y3, 1))./lag(y3, 1), 2); % daily P&L of the strategy
ret=pnl./sum(abs(lag(positions, 1)), 2); % return is P&L divided by gross market value of portfolio
ret(isnan(ret)) =0;


ret1(1)=100;
for i=2:length(ret)
ret1(i)=ret1(i-1)*(1+ret(i));
end

maxdd=maxdrawdown(ret1);
xlswrite('pair_output_updated.xlsx',ret1','Sheet10');

figure;
plot((cumprod(1+ret)-1)*100); % Cumulative compounded return

fprintf(1, 'APR=%f Sharpe=%f  Q/R=%f Beta=%f alpha=%f\n', prod(1+ret).^(252/length(ret))-1, sqrt(252)*mean(ret)/std(ret),x, beta(1,478),beta(2,478));



