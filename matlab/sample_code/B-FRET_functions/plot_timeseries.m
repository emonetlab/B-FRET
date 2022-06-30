function plot_timeseries(data,obs_pred,FRET_sgnl,save_dir)

%TF = isfield(obs_pred,'MCMC');
TF2 = isfield(data,'E'); %Whetehr or not it's an artificial data
cd(save_dir)
cd('./sanity_check')

%% data and computed timeseries
IDD = data.IDD; 
IDA = data.IDA;
IAA = data.IAA;
IDD_noise_sd = data.IDD_noise_sd;
IDA_noise_sd = data.IDA_noise_sd;
IAA_noise_sd = data.IAA_noise_sd;
tDD = data.tDD;
tAA = data.tAA;

if TF2 
    E = data.E; % True E-FRET for an artificial data 
end

% predictions of the observables 
IDD_pred = obs_pred.IDD;
IDA_pred = obs_pred.IDA;
IAA_pred = obs_pred.IAA;

%FRET signals
E_med = FRET_sgnl.E_med;
E_0p13 = FRET_sgnl.E_0p13;
E_2p27 = FRET_sgnl.E_2p27;
E_15p87 = FRET_sgnl.E_15p87;
E_84p13 = FRET_sgnl.E_84p13;
E_97p73 = FRET_sgnl.E_97p73;
E_99p87 = FRET_sgnl.E_99p87;
    
%% Plotting observables
FH = figure('Position',[50 50 600 600]);hold on;
FH.Renderer = 'painters';
col1 = [1 0.3 0.3];
lw = 1;
l1 = plot(tDD,IDD,'c-','LineWidth',lw );
l2 = plot(tAA,IAA,'m-','LineWidth',lw );
l3 = plot(tDD,IDA,'g-','LineWidth',lw );

jbfill(tDD,IDD_pred + 2*IDD_noise_sd, IDD_pred - 2*IDD_noise_sd,'k','None',0,0.2);
l4 = plot(tDD,IDD_pred,'Color',col1,'LineWidth',lw );
jbfill(tAA,interp1(tDD,IAA_pred,tAA) + 2*IAA_noise_sd, interp1(tDD,IAA_pred,tAA) - 2*IAA_noise_sd,'k','None',0,0.2);
plot(tDD,IAA_pred,'Color',col1,'LineWidth',lw );
jbfill(tDD,IDA_pred + 2*IDA_noise_sd, IDA_pred - 2*IDA_noise_sd,'k','None',0,0.2);
plot(tDD,IDA_pred,'Color',col1,'LineWidth',lw );

xlabel('Time (sec)');ylabel('Intensities');

lh = legend([l1 l2 l3 l4],'{\itI_{DD}}','{\itI_{AA}}','{\itI_{DA}}','pred.');
set(lh,'Color','None','Box','Off');
set(gcf,'Visible','on')
saveas(gcf,'observables','fig')
%saveas(gcf,'observables','tif')

%% Plotting B-FRET

FH = figure('Position',[50 50 600 300]);hold on;
FH.Renderer = 'painters';
hold on;
col_CI = [0.7 0.7 0.7];
lw1 = 2;
l1 = plot(tDD,E_med,'Color','b','LineWidth',lw1);
 plot(tDD,E_15p87,'Color',col_CI);
 plot(tDD,E_84p13,'Color',col_CI);
 plot(tDD,E_2p27,'Color',col_CI);
 plot(tDD,E_97p73,'Color',col_CI);
if TF2 %If it's artificial data
    l2 = plot(tDD,E,'Color','r','LineWidth',1);
    lh = legend([l2 l1],'True','Median');
else
    lh = legend([l1],'Median');
end

set(lh,'Color','None','Box','Off');
xlabel('Time (sec)');ylabel('E = \chi/[D]_T');
set(gcf,'Visible','on')
saveas(gcf,'E','fig')
%saveas(gcf,'E','tif')

cd(save_dir)
    
end


