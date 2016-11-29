function mutationChildren = mutation(parents,options,GenomeLength,FitnessFcn,state,thisScore,thisPopulation,mutationRate)
%MUTATION mutate childrend accordinlgy
%   MUTATIONCHILDREN = MUTATIONUNIFORM(PARENTS,OPTIONS,GENOMELENGTH,...
%                      FITNESSFCN,STATE,THISSCORE,THISPOPULATION, ...
%                      MUTATIONRATE)
%   Creates the mutated children using
%   uniform mutations at multiple points. 
%% Error Checking/Default Values
if nargin < 8 || isempty(mutationRate)
    mutationRate = 0.02; % default mutation rate
end

if ~strcmpi(options.PopulationType,'bitString')
    error('MUTATION:NotBitStringType','PopulationType must be a bitString')
end

mutationChildren = false(length(parents),GenomeLength);

%% Select Parents to Mutate 
for i=1:length(parents)
    child = thisPopulation(parents(i),:);
    valid = true;
    while valid
        mutationPoints = find(rand(1,length(child)) < mutationRate);
        child(mutationPoints) = ~child(mutationPoints);
        valid = ~validRule(child);
    end
    mutationChildren(i,:) = child;
end