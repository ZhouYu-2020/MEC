
all = importfile(".\all.xlsx", "Sheet1", [2, 9]);
mec = importfile(".\mec.xlsx", "Sheet1", [2, 9]);
local = importfile(".\local.xlsx", "Sheet1", [2, 9]);
all_5gHZ = importfile(".\all_5gHZ.xlsx", "Sheet1", [2, 9]);
all_8gHZ = importfile(".\all_8gHZ.xlsx", "Sheet1", [2, 9]);
all_13gHZ = importfile(".\all_13gHZ.xlsx", "Sheet1", [2, 9]);
%mec_13ghz = importfile(".\mec-13ghz.xlsx", "Sheet1", [2, 9]);
%%
%figure
%subplot(1,2,1)
Step = [5:5:40];
plot(Step,local,'-^')
hold on
plot(Step,mec,'-*')
hold on
plot(Step,all_5gHZ,'-+')
hold on
plot(Step,all_8gHZ,'-h')
hold on
plot(Step,all,'-o')
hold on
plot(Step,all_13gHZ,'-d')
hold on
%axis( [10 45 -1 15] )
xlabel('the number of UEs ')
ylabel(' the total of system energry  comsumption (J)')
legend('Only-Local(Fser=10GHz)', '  Only-MEC(Fser=10GHz)',   ' Proposed(Fser=5GHz)', ' Proposed (Fser=8GHz)','  Proposed (Fser=10GHz)', '  Proposed£¨Fser=13GHz£©')
%figure


