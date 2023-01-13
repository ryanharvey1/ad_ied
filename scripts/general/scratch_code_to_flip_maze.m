
% 
% files = dir('X:\AD_sessions\**\*session.mat');
% 
% 
% general_behavior_file('basepath',{files.folder})
% 
% 
% general_behavior_file('force_run',true,'force_overwrite',true)

%%

load('AZ18_221124_sess8.animal.behavior.mat')

behavior.position.x = behavior.position.x * -1;
behavior.position.x = behavior.position.x + abs(min(behavior.position.x));

behavior.position.y = behavior.position.y * -1;
behavior.position.y = behavior.position.y + abs(min(behavior.position.y));

figure;plot(behavior.position.x,behavior.position.y)

save('AZ18_221124_sess8.animal.behavior.mat','behavior')


figure;plot(behavior.position.x,behavior.position.y)
hold on
for i = 1:10:length(behavior.position.x)
    scatter(behavior.position.x(i),behavior.position.y(i))
end