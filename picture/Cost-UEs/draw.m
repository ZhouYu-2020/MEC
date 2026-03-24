
all = importfile(".\all.xlsx", "Sheet1", [2, 9]);
mec = importfile(".\mec.xlsx", "Sheet1", [2, 9]);
local = importfile(".\local.xlsx", "Sheet1", [2, 9]);
%all_5gHZ = importfile(".\all_5gHZ.xlsx", "Sheet1", [2, 9]);
all_8gHZ = importfile(".\all_8gHZ.xlsx", "Sheet1", [2, 9]);
all_13gHZ = importfile(".\all_13gHZ.xlsx", "Sheet1", [2, 9]);

%%
figure
%subplot(1,2,1)
LineWidth = 1.5;
Step = [5:5:40];
plot(Step,local,'-^','LineWidth',LineWidth)
hold on
plot(Step,mec,'-*','LineWidth',LineWidth)
hold on
%plot(Step,all_5gHZ,'-+')
%hold on
plot(Step,all,'-o','LineWidth',LineWidth)
hold on
plot(Step,all_8gHZ,'-h','LineWidth',LineWidth)
hold on
plot(Step,all_13gHZ,'-d','LineWidth',LineWidth)
hold on
grid on;
%grid minor;
%axis( [10 45 -1 15] )
xlabel('the number of UEs ')
ylabel(' the total of system energry  comsumption (J)')
legend(' Non MEC', ' Stand alone MEC', ' Proposed EVT learning',' Proposed EVT learning (Fser=8GHz)', ' Proposed EVT learning (Fser=13GHz)')
%figure


