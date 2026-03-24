% POTpara
length = 1001
para0 = importfile(".\para0.xlsx", "Sheet1", [2, length]);
para1 = importfile(".\para1.xlsx", "Sheet1", [2, length]);


%%

Step = [1:1:1000];


windowSize = 10;
yy=filter(ones(1,windowSize)/windowSize,1,para0);
xx = filter(ones(1,windowSize)/windowSize,1,para1);

[hAx,hLine1,hLine2] = plotyy(Step,xx,yy,'r');
grid on;
%grid minor;

xlabel('Episode')
%ylabel(hAx(1),'Approximated GPD scale parameter value \sigma_{i}') ;
%ylabel(hAx(2),'Approximated GPD shape parameter value \xi_{i}');

ylabel(hAx(1),'Approximated GPD scale parameter value ') ;
ylabel(hAx(2),'Approximated GPD shape parameter value ');

%legend('local only','wolf phc ','mec only','wolf-phc-dl2-dh5','wolf-phc-dl1-dh4')
%legend('scale parameter','shape parameter')
%figure
