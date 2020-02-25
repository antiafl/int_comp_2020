function [model] = train_linearDisc_Kfold_p1(k,INPUTS,OUTPUTS,discrType,CV)

for i = 1:k
    trIdx = CV.training(i);   
    INPUTTRAIN=INPUTS(trIdx,:);
    DTRAIN=OUTPUTS(trIdx,:);

    model{i} = fitcdiscr(INPUTTRAIN, DTRAIN, 'DiscrimType', discrType);
end
