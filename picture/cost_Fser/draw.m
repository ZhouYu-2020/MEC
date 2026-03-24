
% å›?3
length = 7  ;
all = importfile(".\all.xlsx", "Sheet1", [2,length]);
mec = importfile(".\mec.xlsx", "Sheet1", [2, length]);
local = importfile(".\local.xlsx", "Sheet1", [2,length]);
all_10_UEs = importfile(".\all_10_UEs.xlsx", "Sheet1", [2,length]);
all_20_UEs = importfile(".\all_20_UEs.xlsx", "Sheet1", [2, length]);
%all_25_UEs = importfile(".\all_25_UEs.xlsx", "Sheet1", [2, length]);

%%

Step = [4:2:14 ];
figure;
LineWidth = 1.5;
plot(Step,local,'-^','LineWidth',LineWidth)
hold on
plot(Step,mec,'-*','LineWidth',LineWidth)
hold on

plot(Step,all_10_UEs,'-+','LineWidth',LineWidth)
hold on
plot(Step,all,'-o','LineWidth',LineWidth)
hold on
plot(Step,all_20_UEs,'-h','LineWidth',LineWidth)
hold on
%plot(Step,all_25_UEs,'-d')
%hold on
%axis( [10 45 -1 15] )
grid on;
%grid minor;
xlabel('the capacity of MEC server (GHz) ')
ylabel(' the total of system energry  comsumption (J)')
legend('Non MEC(UEs=15)','Stand alone MEC(UEs=15)','Proposed EVT learning (UEs=10)','Proposed EVT learning (UEs=15)','Proposed EVT learning (UEs=20)')

%legend('wolf phc ','offloading only','wolf-phc-dl2-dh5','wolf-phc-dl1-dh4')
%figure
