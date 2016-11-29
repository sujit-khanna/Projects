function s = tradeSignal(pop,ind)
%TRADSIGNAL generates the trading signal from population shown below
%      0   |  0     0 |   0  |   1   0  |   1  |  1    1     1 |
%     IND1 |   AND    | IND2 |    OR    | IND3 |IND1  IND2 IND3|
%% BitString Length
L = size(pop,2);

%% Calculate Indicators
% Length = 4*NoIndicators - 2
I = (L+2)/4;

% indicator locations
indLoc = 1:3:L-I;
%% Generate Signal
s = zeros(size(ind,1),size(pop,1));
for r = 1:size(pop,1)
    ind2use = logical(pop(r,L-I+1:end));
    idx = indLoc(ind2use);
    for i = 1:length(idx)-1
        if i == 1
            idxstr = ['idx(',num2str(i),')'];
            % indicator 1
            A = eval(['(ind(:,1) == pop(r,',idxstr,'));']);
            
        else
            A = C;
        end
        % indicator 2
        idxstr = ['idx(',num2str(i+1),')'];
        B = eval(['(ind(:,',num2str(i+1),') == pop(r,',idxstr,'));']);
        % connector
        conValue = bin2dec(num2str(pop(r,(1:2)+idx(i))));
        switch conValue
            case 0
                cv = 'and';   %and
            case 1
                cv = 'or';   %or
            case 2
                cv = 'xor';
        end    
        C = eval([cv,'(A,B)']);
    end
    s(:,r) = C;
end
