function xoverKids  = crossover(parents,options,GenomeLength,FitnessFcn,unused,thisPopulation)
%CROSSOVER generates kids from parents
%   KID = CROSSOVER(PARENTS,OPTIONS,GENOMELENGTH,FITNESSFCN,UNUSED,...
%                   THISPOPULATION)
%
iterLimit = 1000;
% Number of children to produce
nKids = length(parents)/2;
% Allocate space for the kids
xoverKids = false(nKids,GenomeLength);

% valid connector/indicator locations
con = locateConnectors(thisPopulation(1,:));
index = 1;
for i=1:nKids
    % get parents
    r1 = parents(index);
    index = index+1;
    r2 = parents(index);
    index = index+1;
    
    valid = true;
    iter = 1;
    while valid
        % Randomly select one location for crossover from parents
        xoverPoints = randi(GenomeLength);
        % check if points are same type (connector or indicator)
        isCon = ismember(xoverPoints,con);
        if isCon % connectors
            xoverPoints(2) = con(ismember(con,[xoverPoints(1)-1 xoverPoints(1)+1]));
            xoverPoints = sort(xoverPoints);
        end
        
        % index for head/tail part
        head = 1:xoverPoints(end);
        tail = xoverPoints(end)+1:GenomeLength;
        
        % Split parents at connectors to create two new children
        kid(1,:) = [thisPopulation(r1,head) thisPopulation(r2,tail)];
        kid(2,:) = [thisPopulation(r2,head) thisPopulation(r1,tail)];
        
        valid = prod(double(~validRule(kid)));
        iter = iter+1;
        if iter > iterLimit
            error('CROSSOVER:ITERLIMIT', 'Iteration Limit Reached!')
        end
    end % while loop
    
    % randomly select one of the possible offspring to be used
    xoverKids(i,:) = kid(randi([1 2]),:);
end % kid loop
