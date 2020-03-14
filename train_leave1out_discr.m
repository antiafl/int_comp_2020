function [model,INPUTTRAIN,DTRAIN] = train_leave1out_discr(N,INPUTS,OUTPUTS,discrType,CV)

for i = 1:N
    trIdx = CV.training(i);   
    INPUTTRAIN{i}=INPUTS(trIdx,:);
    DTRAIN{i}=OUTPUTS(trIdx,:); 
    model{i} = fitcdiscr(INPUTTRAIN{i}, DTRAIN{i}, 'DiscrimType', discrType);    
end