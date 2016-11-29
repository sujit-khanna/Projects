function [conIndex indIndex] = locateConnectors(rule);
%LOCATECONNECTORS finds locations of connectors in RULE
%   CONINDEX = LOCATECONNECTORS(RULE) returns the indicies into RULE that
%   are connectors
%% Extract indicators from rule length
if isvector(rule)
    L = length(rule); % column/row vector
else
    L = size(rule,2); % array
end

%% Calculate Indicators
% Length = 4*NoIndicators - 2
I = (L+2)/4;

%% Connector Locations
conIndex = [];
for b = 2:3:L-I
    conIndex = [conIndex b b+1];
end
%% Indicator Locations
indIndex = 1:L;
indIndex = indIndex(~ismember(indIndex,conIndex));

