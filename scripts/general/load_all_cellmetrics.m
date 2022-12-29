% load_all_cellmetrics
df = readtable("D:\github\ad_ied\data\sessions.csv",'Delimiter',',');


basenames = [];
for i = 1:length(df.basepath)
    basenames{i} = basenameFromBasepath(df.basepath{i});
end

% load all cell metrics
cell_metrics = loadCellMetricsBatch('basepaths',df.basepath,'basenames',basenames);

% pull up gui to inspect all units in your project
cell_metrics = CellExplorer('metrics',cell_metrics);