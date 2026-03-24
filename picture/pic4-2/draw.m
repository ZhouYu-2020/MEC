length = 7

all_twait = importfile(".\all+twait.xlsx", "Sheet1", [2, length]);
all_twai_30_UEs = importfile(".\all+twait+30_UEs.xlsx", "Sheet1", [2, length]);
local = importfile(".\local.xlsx", "Sheet1", [2, length]); 
local_30 = importfile(".\local+30_UEs.xlsx", "Sheet1", [2, length]); 
mec_twait = importfile(".\mec+twait.xlsx", "Sheet1", [2, length]); 


Step = [0.05:0.02:0.15 ];

plot(Step,all_twai_30_UEs,'-*')
hold on
plot(Step,local,'-o')
hold on
plot(Step,local_30,'-o')
hold on
plot(Step,mec_twait,'-o')
hold on
plot(Step,all_twait,'-^')
hold on

xlabel('The different \\t^{max} (s)')
ylabel('Sum cost (J)')
legend('Non MEC','Non MEC of 30 UEs,','Stand alone MEC','EVT learning ','EVT learning for 30UEs')
%figure



