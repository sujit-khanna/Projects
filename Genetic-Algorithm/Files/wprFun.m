function sh = wprFun(x,data,scaling,cost)
row = size(x,1);
sh = zeros(row,1);
parfor i = 1:row
    [~,~,sh(i)] = wpr(data,x(i,1),scaling,cost);
end