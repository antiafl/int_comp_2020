function [model] = train_Kfold_discr(k,INPUTS,OUTPUTS,discrType,CV)

for i = 1:k
    trIdx = CV.training(i);   
    INPUTTRAIN=INPUTS(trIdx,:);
    DTRAIN=OUTPUTS(trIdx,:);
    model{i} = fitcdiscr(INPUTTRAIN, DTRAIN, 'DiscrimType', discrType);
end
