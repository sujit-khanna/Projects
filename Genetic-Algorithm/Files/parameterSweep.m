function [respmax,varmax,resp,var] = parameterSweep(fun,range)
%PARAMETERSWEEP performs a parameters sweep for a given function
%% Check inputs
if nargin == 0
    example(1)
else
    %% Generate expression for ndgrid
    N = length(range);
    if N == 1
        var = range;
    else
        in = ''; out = '';
        for i = 1:N
            in  = [in,'range{',num2str(i),'},'];
            out = [out,'var{',num2str(i),'},'];
        end
        in(end)  = []; % remove last commas
        out(end) = [];
        
        %% Evaluate ndgrid
        eval( ['[',out,'] = ndgrid(',in,');'] );
    end
    %% Perform parameter sweep
    sz = size(var{1});
    for i = 1:N
        var{i} = var{i}(:);
    end
    resp = fun(cell2mat(var));
    
    %% Find maximum value and location
    [respmax,idx]   = max(resp);
    for i = 1:N
        varmax(i) = var{i}(idx);
    end
    
    %% Reshape output only if requested
    if nargout > 2
        resp = reshape(resp,sz);
        for i = 1:N
            var{i} = reshape(var{i},sz);
        end
    end %if
    
end %if

%% Examples
function example(ex)
for e = 1:length(ex)
    for e = 1:length(ex)
        switch ex(e)
            case 1
                figure(1), clf
                range = {-3:0.1:3, -3:0.2:3};  % range of x and y variables
                fun = @(x) deal( peaks(x(:,1),x(:,2)) ); % peaks as a function handle
                [respmax,varmax,resp,var] = parameterSweep(fun,range);
                surf(var{1},var{2},resp)
                hold on, grid on
                plot3(varmax(1),varmax(2),respmax,...
                    'MarkerFaceColor','k', 'MarkerEdgeColor','k',...
                    'Marker','pentagram', 'LineStyle','none',...
                    'MarkerSize',20, 'Color','k');
                hold off
                xlabel('x'),ylabel('y')
                legend('Surface','Max Value','Location','NorthOutside')
        end %switch
    end %for
end %for


