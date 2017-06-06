clear all;
close all;
clc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Test mrf_tool: 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%% Manual settings:
%%% Fill in the appropriate numbers:
% Choose z_axis to be plotted ( multiple optioins possible)
z_col = [14,18];
x_data_row = 32;  %data for x values
sample_row = 70;  % row in parameters in which the sample number stands
% Path to data:
path = '/tmp/temp/eval_scenenet_two/';
data_set = 'scenenet';
x_axis_name = 'Discontinuity Threshold';
name = 'dis_thresh';

% parameter loop sizes:
i_size = 86;         % inner loop
m_size = 4;          % middle loop
o_size = 2;          % outer loop


data = importdata([path '/eval_init.log'],',');
parameters = data.textdata';
linesize = 2;
save_figures = 0;   % saves figures to path

m_total = cellstr(parameters);
[c,r] = size(data.data);
for i=1:1:c
    d = data.data(i,:)';
    m_total(:,i+1) = num2cell(d);
    clear d;
end


one_run_size = i_size*m_size*o_size
offset = mod(c,i_size*m_size*o_size);
m_data_total = cell2mat(m_total(:,2:end-offset));
c = c-offset;
start = 0;

num_of_samples = max(m_data_total(sample_row,:))+1;  %% for planes evaluation set to 1;
length_samle_run = i_size*o_size*m_size;;
length_of_each_data = i_size*m_size;

fig_nr = 1;

%%  run plots
for p=1:1:length(z_col)
    x_axis_data =  m_data_total(x_data_row,1:length_of_each_data);
   
    z_data_dingler = zeros(m_size,i_size);
    z_data_normal = zeros(m_size,i_size);

%loop through each sample run
    for sample_nr = 1:1:num_of_samples
        start_col = (sample_nr-1)*length_samle_run +1;
        end_col = sample_nr*length_samle_run;
        m_data = m_data_total(:,start_col:end_col);

        z_normal = m_data(z_col(p),1:length_of_each_data);
        z_dingler = m_data(z_col(p),(length_of_each_data+1):(2*length_of_each_data));
        
        for m_index=1:1:m_size
            st = (m_index-1)*i_size+1;
            en = (m_index*i_size);
           
            z_data_normal(m_index,:) =  z_data_normal(m_index,:)+ z_normal(st:en);
            z_data_dingler(m_index,:) =  z_data_dingler(m_index,:)+ z_dingler(st:en);
            
        end
    end
     z_data_normal(:,:) =  z_data_normal(:,:)./num_of_samples; 
     z_data_dingler(:,:) =  z_data_dingler(:,:)./num_of_samples;
  
    
    
z_axis_name = parameters(z_col(p))';
color_diebel =[0 0 0];
color_ours = [ 0    0.6000    0.2000];

x_data = x_axis_data(1:i_size);

%% plot ours
fig_ours = figure(fig_nr);
fig_nr = fig_nr +1;
h = zeros(4);

   h(1) = plot(x_data,z_data_normal(1,:),'--*','Color','black'); hold on;
   h(2) = plot(x_data,z_data_normal(2,:),'--*','Color','blue'); hold on;
   h(3) = plot(x_data,z_data_normal(3,:),'--*','Color','g'); hold on;
   h(4) = plot(x_data,z_data_normal(4,:),'--*','Color','r'); hold on;
  % h(m_index + m_size) = plot(x_data,z_data_dingler(m_index,:),'--*','Color',color_diebel); hold on;
leg= legend([h(1),h(2),h(3),h(4)],{'None','Step','Linear','Exponential'});
leg.Location = 'northeastoutside';
title('Ours');
xlabel([strrep(x_axis_name,'_',' ')]);
ylabel([strrep(z_axis_name,'_',' ' )]);

leg.Location = 'northeastoutside';
if save_figures ==1
    mkdir([path '/Figures/']);
    savefig(fig_ours,[path '/Figures/Ours_' data_set '_' name '_' cell2mat(z_axis_name) '.fig']);
    saveas(fig_ours,[path '/Figures/Ours_' data_set '_' name '_' cell2mat(z_axis_name) '.eps'],'epsc');
end

set(gca,'YLim',[0.04 0.16]);
%% plot dingler
fig_ours = figure(fig_nr);
fig_nr = fig_nr +1;
h = zeros(4);

   h(1) = plot(x_data,z_data_dingler(1,:),'--*','Color','black'); hold on;
   h(2) = plot(x_data,z_data_dingler(2,:),'--*','Color','blue'); hold on;
   h(3) = plot(x_data,z_data_dingler(3,:),'--*','Color','g'); hold on;
   h(4) = plot(x_data,z_data_dingler(4,:),'--*','Color','r'); hold on;
  % h(m_index + m_size) = plot(x_data,z_data_dingler(m_index,:),'--*','Color',color_diebel); hold on;
leg= legend([h(1),h(2),h(3),h(4)],{'None','Step','Linear','Exponential'});
leg.Location = 'northeastoutside';
title('Diebel');
xlabel([strrep(x_axis_name,'_',' ')]);
ylabel([strrep(z_axis_name,'_',' ' )]);

set(gca,'YLim',[0.04 0.16]);
leg.Location = 'northeastoutside';

if save_figures ==1
    mkdir([path '/Figures/']);
    savefig(fig_ours,[path '/Figures/Diebel_' data_set '_' name '_' cell2mat(z_axis_name) '.fig']);
    saveas(fig_ours,[path '/Figures/Diebel_' data_set '_' name '_' cell2mat(z_axis_name) '.eps'],'epsc');
end
end
