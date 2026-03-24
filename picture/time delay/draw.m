length = 7

%all_twait = importfile('.\all+twait.xlsx', 'Sheet1', [2, length]);
%all_twait_30_UEs = importfile('.\all+twait+30_UEs.xlsx', 'Sheet1', [2, length]);
%local = importfile('.\local.xlsx', 'Sheet1', [2, length]); 
%local_30 = importfile('.\local+30_UEs.xlsx', 'Sheet1', [2, length]); 
%mec_twait = importfile('.\mec+twait.xlsx', 'Sheet1', [2, length]); 
load matlab.mat


Step = [5,10,20,30,40,50]
LineWidth = 1.5;
plot(Step,local,'-o','LineWidth',LineWidth)
hold on
plot(Step,local_30,'--o','LineWidth',LineWidth)
hold on
plot(Step,mec_twait,'-x','LineWidth',LineWidth)
hold on
plot(Step,mec_twait_30UEs,'--x','LineWidth',LineWidth)
hold on

plot(Step,all_twait,'-*','LineWidth',LineWidth)
hold on
plot(Step,all_twait_30UEs,'--*','LineWidth',LineWidth)
hold on
%axis( [5 55 0 150 ] )
grid on;
%grid minor;

xlabel('the time delay threshold t^{\it{max}} (ms)')
ylabel('the total of system energry comsumption (J)')
legend('Non MEC (UEs=15)','Non MEC (UEs=30)','Stand alone MEC (UEs=15)','Stand alone MEC (UEs=30)','Proposed EVT learning (UEs=15)','Proposed EVT learning (UEs=30)')
hold on;
%legend()

%figure



