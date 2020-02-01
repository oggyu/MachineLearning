clear 
clc 

input = 2 ; 
output = 1 ;
hidden = 16 ; 
w1 = rand(hidden,2);%.*2-1;
w2 = rand(hidden,1);%.*2-1; 

iter = 0; 
regRate = 0 ; 

inputOutput = [0.1 0.1 0.2;...
              0.1 0.2 0.3;...
              0.1 0.3 0.4;...
              0.2 0.1 0.3;...
              0.2 0.2 0.4;...
              0.2 0.3 0.5;...
              0.3 0.1 0.4;...
              0.3 0.2 0.5;...
              0.3 0.3 0.6;...
              0.4 0.4 0.8;...
              0.4 0.5 0.9;...
              0.3 0.5 0.8;...
              0.2 0.5 0.7;...
              0.1 0.5 0.6];
 
learningRate = 0.1 ;

changeW1 = zeros(hidden,2); 
changeW2 = zeros(hidden,1); 



while(iter < 70000) 
sumOfChangesW1 = zeros(hidden,2);
sumOfChangesW2 = zeros(hidden,1);
    %feed forward check output
   for i = 1:size(inputOutput,1)
        z1 = w1*[inputOutput(i,1);inputOutput(i,2)];
        a1 = 1./(1+exp(-z1));
        z2 = sum(w2 .* a1);
        a2 = 1./(1+exp(-z2));                         % a2 => output 
        
        outputErr = inputOutput(i,3) - a2 ; 
        
        delta2 = -(a2 - inputOutput(i,3)).*a2.*(1-a2);        %  - d(Err)/d(weight) == delta
        delta1 = delta2 .* w2 .* a1.*(1-a1); 
        
%         changeW2 = -learningRate .* -delta2.*a1; 
%         changeW1(:,1) = -learningRate .* -delta1.*inputOutput(i,1) ;
%         changeW1(:,2) = -learningRate .* -delta1.*inputOutput(i,2) ; 
        
         changeW2 = -learningRate .* ( -delta2.*a1 + w2.*regRate); 
        changeW1(:,1) = -learningRate .* ( -delta1.*inputOutput(i,1) + w1(:,1).*regRate) ;
        changeW1(:,2) = -learningRate .* ( -delta1.*inputOutput(i,2) + w1(:,2).*regRate) ; 
        
        sumOfChangesW1 = sumOfChangesW1 + changeW1;
        sumOfChangesW2 = sumOfChangesW2 + changeW2; 
%         w1 = w1 + changeW1;
%         w2 = w2 + changeW2;
        
        
   end
   w1 = w1 + (sumOfChangesW1./size(inputOutput,1));
   w2 = w2 + (sumOfChangesW2./size(inputOutput,1));
   iter = iter + 1 ;
    
end

save('weights','w1','w2')
% 
%  changeW2 = -learningRate .* ( -delta2.*a1 + w2.*regRate); 
%         changeW1(:,1) = -learningRate .* ( -delta1.*inputOutput(i,1) + w1(:,1).*regRate) ;
%         changeW1(:,2) = -learningRate .* ( -delta1.*inputOutput(i,2) + w1(:,2).*regRate) ; 