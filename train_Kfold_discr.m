function [model,INPUTTRAIN,DTRAIN] = train_Kfold_discr(k,INPUTS,OUTPUTS,discrType,CV)

for i = 1:k
    trIdx = CV.training(i);   
    INPUTTRAIN{i}=INPUTS(trIdx,:);
    DTRAIN{i}=OUTPUTS(trIdx,:);
    model{i} = fitcdiscr(INPUTTRAIN{i}, DTRAIN{i}, 'DiscrimType', discrType);
end
