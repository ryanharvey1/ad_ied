% this function solves these issue:
%       epochs have lfp saved within which slows down loading
%       the files are saved with a format unreadable by python
%       the events are stored within different variables

df = readtable("D:\github\ad_ied\data\sessions.csv",'Delimiter',',');


for i = 1:length(df.basepath)
    disp(df.basepath{i})
    run(df.basepath{i})
end

function run(basepath)
basename = basenameFromBasepath(basepath);

load(fullfile(basepath,[basename,'.IED.events.mat']))

if exist('Epi','var')
    IED = Epi;
end

if exist('Epi_Events','var')
    IED = Epi_Events;
end

if exist('IED','var')
    fields = {'lfp','signal'};
    try
        IED = rmfield(IED,fields);
    catch
        save(fullfile(basepath,[basename,'.IED.events.mat']),'IED')
        return
    end
    save(fullfile(basepath,[basename,'.IED.events.mat']),'IED')
    return
end

warning(basepath)
error('not Epi or IED')
end