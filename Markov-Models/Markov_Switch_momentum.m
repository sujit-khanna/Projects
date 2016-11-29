clc;
clear all;
close all;


%[a b]=xlsread('NSE Hist data 2005-2011.xlsx','sheet2');
%addpath('m_Files'); % add 'm_Files' folder to the search path
%addpath('data_Files');
%[a b]=xlsread('final_data.xlsx','sheet1');

[a b]=xlsread('NSEdata.xlsx','sheet1');
prices=a(:,1);
logRet=importdata('logreturns.txt');  % load some Data.

dep=logRet(:,1);   % Defining dependent variable from .mat file
%dep=a(2:1984,3);
constVec=ones(length(dep),1);       % Defining a constant vector in mean equation (just an example of how to do it)
indep = ones(size(logRet));         % A dummy explanatory variable     % Defining some explanatory variables
k=2;                                % Number of States
S=[1 1];                            % Defining which parts of the equation will switch states (column 1 and variance only)
advOpt.distrib='Normal';            % The Distribution assumption ('Normal', 't' or 'GED')
advOpt.std_method=1;                % Defining the method for calculation of standard errors. See pdf file for more details

[Spec_Out]=MarkovFit(dep,indep,k,S,advOpt); % Estimating the model
OP_TransitProb=Spec_Out.smoothProb;

%%Logic to generate signal here
state1=OP_TransitProb(:,1);
state2=OP_TransitProb(:,2);

%%market Regimes
idx1=state2>0.80;

%Regime Switching Momentum here
%6month/1month signal logic
for i=126:length(prices)
   ret6(i)=(prices(i)/prices(i-125))-1;
   ret3(i)=(prices(i)/prices(i-66))-1;
   ret1(i)=(prices(i)/prices(i-22))-1;
   retday(i)=(prices(i)/prices(i-1))-1;
end

   
%calcuate end points as months in matlab
signal(125)=0
for i=126:length(prices)

    if mod(i,22)==0 & idx1(i)==0
       signal(i)=sign(ret6(i));
    elseif mod(i,22)==0 & idx1(i)==1
        signal(i)=sign(ret1(i));
    else 
        signal(i)=signal(i-1);
    end 
        
end

strat_idx(125)=100
for i=126:length(prices)
strat_ret(i)=signal(i-1)*retday(i);
strat_idx(i)=strat_idx(i-1)*(1+strat_ret(i));
end

figure;
plot(strat_idx(126:length(strat_idx))); % Cumulative compounded return
