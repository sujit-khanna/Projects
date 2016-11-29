function tf = validRule(rule)
%VALIDRULE determines if the rule is valid
%% Extract indicators from rule length
if isvector(rule)
    L = length(rule); % column/row vector
else
    L = size(rule,2); % array
end
P = size(rule,1);
%% Calculate Indicators
% Length = 4*NoIndicators - 2
I = (L+2)/4;

%% Check for Valid Connectors
tf = false(P,1);
for i = 1:P
    for b = 2:3:L-I
        if sum(rule(i,[b b+1])) == 2
            tf(i) = false;
            break
        else
            tf(i) = true;
        end
    end
end
