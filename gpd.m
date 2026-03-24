


function res = gpd(y,threshold )

[F,xi] = ecdf(y);
paramEsts = gpfit(y);
kHat      = paramEsts(1);
sigmaHat = paramEsts(2);

yi = gpcdf(xi,kHat,sigmaHat);




probability = 0 ;
[length,r] = size(xi);

if threshold <  xi(1) 
    probability = 0;
end 
if threshold >=  xi( length )
    probability = 1;
end 


if  (xi(1) <=  threshold)  & ( threshold < xi(length)  ) 
    
    for  i=2:length
     if (xi(i-1) <= threshold  )  & ( threshold  <  xi(i) ) 
        [k,b] = forkb(xi(i-1),yi(i-1),xi(i),yi(i));
        probability =  (k*threshold + b);
      
        break;
     end
    end
end
for  i=1:length
    F(i) = 1 - F(i);
    yi(i) = 1 - yi(i);
end

isPicture = 0;
if isPicture > 0
    plot(xi,yi,'r');% 拟合出来的线
    hold on;
    stairs(xi,F,'b');  % 输入的数据
    hold on;
    %plot(threshold,probability,'ro')
    hold on
    xlabel('Queue length Q_{i}(t)  /bits q_{0}=3.76*10^7 bits')
    ylabel('CCDF')
    legend('Approximated GPD','Numerial results')
    %saveas(gcf,['.\gpd\',num2str(length),'.fig'])
    saveas(gcf,['.\gpd\',num2str(length),'.png'])
    hold off
    xlswrite(['.\gpd\',num2str(length),'.xlsx']  , [xi,F,yi])
end



%probability = 0.55
res = [kHat,sigmaHat ,probability] ;
end

function [k,b] = forkb(x1,y1,x2,y2)
    k = (y2-y1)/(x2-x1);
    b = y1 - x1*k;
end

