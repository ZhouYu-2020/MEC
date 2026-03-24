length = 7

all = importfile(".\all.xlsx", "Sheet1", [2, length]);
local = importfile(".\local.xlsx", "Sheet1", [2, length]);
mec = importfile(".\mec.xlsx", "Sheet1", [2, length]); 


Step = [0.05:0.02:0.15 ];
plot(Step,local,'-^')
hold on
plot(Step,mec,'-*')
hold on
plot(Step,all,'-o')
hold on

xlabel('time delay threshold (s)')
ylabel('the total of system energry comsumption (J)')
legend('Only-Local','Only-MEC','Proposed method')
%figure



