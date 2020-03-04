function [model] = train_leave1out_discr(N,INPUTS,OUTPUTS,discrType,CV)

for i = 1:N
    trIdx = CV.training(i);   
    INPUTTRAIN=INPUTS(trIdx,:);
    DTRAIN=OUTPUTS(trIdx,:); 
    model{i} = fitcdiscr(INPUTTRAIN, DTRAIN, 'DiscrimType', discrType);    
end