% CCDF 
all13 = importfile(".\all13.xlsx", "Sheet1", [1, 13]);
mec13 = importfile(".\mec13.xlsx", "Sheet1", [1, 13]); 
plot( all13(:,1) ,all13(:,2) ,'b-+');
hold on;
plot( all13(:,1) ,all13(:,3) ,'r-+');

plot( mec13(:,1) ,mec13(:,2) ,'b-*');
hold on;
plot( mec13(:,1) ,mec13(:,3) ,'r-*');
hold on;
grid on;
%grid minor;
xlabel('Queue length (bits)')
ylabel('CCDF (100%)')
legend('Numerial results with proposed EVT learning','Approximated GPD with proposed EVT learning','Numerial results with stand alone MEC','Approximated GPD with stand alone MEC')
%figure