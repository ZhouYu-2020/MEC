%convergence
mec = importfile(".\mec.xlsx", "Sheet1", [2, 2001]);
only_local = importfile(".\only_local+timedelay+5ge.xlsx", "Sheet1", [2, 2001]);
wolf = importfile(".\wolf.xlsx", "Sheet1", [2, 2001]);
wolf_dl1_dh4 = importfile(".\wolf_dl1_dh4.xlsx", "Sheet1", [2, 2001]);
wolf_dl2_dh5 = importfile(".\wolf_dl2_dh5.xlsx", "Sheet1", [2, 2001]);
%wolf_dl2_dh6 = importfile(".\wolf_dl2_dh6.xlsx", "Sheet1", [2, 2001]);
for  i=1:size(wolf)
    wolf_dl1_dh4(i) = wolf_dl1_dh4(i) -5;
    wolf_dl2_dh5(i) = wolf_dl2_dh5(i) -7;
   
end



%%

Step = [1:1:2000];
windowSize = 10

yy=filter(ones(1,windowSize)/windowSize,1,only_local);
plot(Step,yy)
hold on
windowSize = 1
yy=filter(ones(1,windowSize)/windowSize,1,mec);
plot(Step,yy)
hold on

windowSize = 10;
yy=filter(ones(1,windowSize)/windowSize,1,wolf);
plot(Step,yy)
hold on

%yy=filter(ones(1,windowSize)/windowSize,1,wolf_dl2_dh6);
%plot(Step,yy)
%hold on

yy=filter(ones(1,windowSize)/windowSize,1,wolf_dl2_dh5);
plot(Step,yy)
hold on
yy=filter(ones(1,windowSize)/windowSize,1,wolf_dl1_dh4);
plot(Step,yy)
hold on

grid on;
%grid minor;

axis( [0 2000 -500 120] )

xlabel('Episode')
ylabel('Reward')
%legend('local only','wolf phc ','mec only','wolf-phc-dl2-dh5','wolf-phc-dl1-dh4')
legend('Non MEC',' Stand alone MEC','proposed EVT learning with \delta_{1}=0.002 ,\delta_{h}=0.004','proposed EVT learning with \delta_{1}=0.002 , \delta_{h}=0.005','proposed EVT learning with \delta_{1}=0.001 , \delta_{h}=0.004')
%figure
