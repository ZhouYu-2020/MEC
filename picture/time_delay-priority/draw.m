lp = [416.1,431.2,51.7];
mp = [18.1,27.9,3.4];
hp = [16,120.7,5];
  
subplot(1,3,1) 
width = 1
bar(  [416.1,0,0],width,'r') ;
hold on ;
bar([0,0,51.7],width,'y') ;
bar([0, 431.2],width,'b')  ;
grid on;
xlabel('Lowest priority t^{\it{max}} = 500ms')
ylabel('Average time delay (ms)')
axis( [0 3.5 0 500] )

subplot(1,3,2)
bar([18.1,0,0 ],width,'r');
hold on;
bar([0, 27.9],width,'b');
bar([0,0,3.4],width,'y');
grid on;
xlabel('Medium priority t^{\it{max}} = 50ms')

subplot(1,3,3)
bar([1.6 ,0,0],width,'r');
hold on;
bar([0, 25.7],width,'b');
bar([0,0,3.7],width,'y');
grid on;
xlabel('Highest priority t^{\it{max}} = 5ms')
legend('proposed EVT learning','Non MEC','Stand alone MEC')
