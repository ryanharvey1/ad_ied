

files = dir('X:\AD_sessions\**\*session.mat');

for i = 1:length(files)
    run(files(i).folder)
end

function run(basepath)

    basename = basenameFromBasepath(basepath);

    load(fullfile(basepath,[basename,'.session.mat']),'session');
    
    epochs = [];
    for ep_i = 1:length(session.epochs)
        epochs(ep_i) = isfield(session.epochs{ep_i},'environment');
    end
    if all(epochs)
       return
    end
    if ~exist(fullfile(basepath,[basename,'.Tracking.Behavior.mat']),'file')
       disp([basepath, ' no tracking']) 
       return
    end
    load(fullfile(basepath,[basename,'.Tracking.Behavior.mat']),'tracking')
    load(fullfile(basepath,[basename,'.SleepState.states.mat']),'SleepState')

%     epochs = [];
%     for ep_i = 1:length(session.epochs)
%         epochs = [epochs;[session.epochs{ep_i}.startTime,session.epochs{ep_i}.stopTime]];
%     end
%     PlotIntervals(epochs,'Color',[])

    figure;
    hold on
    plot(tracking.timestamps,tracking.position.x)
    plot(SleepState.idx.timestamps,SleepState.idx.states)

    filenamestruct = dir(fullfile(basepath,[basename,'.lfp']));
    dataTypeNBytes = numel(typecast(cast(0, 'int16'), 'uint8')); % determine number of bytes per sample
    nChannels = session.extracellular.nChannels;
    nSamp = filenamestruct.bytes/(nChannels*dataTypeNBytes);  % Number of samples per channel
    session_start = 0;
    session_stop = nSamp / session.extracellular.srLfp;
    plot([session_start,session_start],[0,20])
    plot([session_stop,session_stop],[0,20])
    
    
    for ep_i = 1:length(session.epochs)
        try
            session.epochs{ep_i}.stopTime;
        catch
            keyboard
        end
        PlotIntervals([session.epochs{ep_i}.startTime,session.epochs{ep_i}.stopTime],...
            'Color',[rand(1),rand(1),rand(1)])
    end
    
    session.general.basePath = basepath;
    gui_session(session);
%     PlotIntervals(tracking.events.subSessions,'Color',[0 0.75 0])
%     PlotIntervals([session_start,session_stop],'Color',[0 0 0.75])
%     xlim([session_start,session_stop])
    close all
end

% behavior.timestamps
% 
%     plot(behavior.timestamps,behavior.position.x)
