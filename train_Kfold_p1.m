function [model] = train_Kfold_p1(k,INPUTS,OUTPUTS,discrType,CV)

for i = 1:k
    trIdx = CV.training(i);   
    INPUTTRAIN=INPUTS(trIdx,:);
    DTRAIN=OUTPUTS(trIdx,:);
    if (strcmp(discrType,'linear') == 1) || (strcmp(discrType,'quadratic') == 1) 
        model{i} = fitcdiscr(INPUTTRAIN, DTRAIN, 'DiscrimType', discrType);
    elseif (strcmp(discrType,'tree') == 1)
        model{i} = fitctree(INPUTTRAIN, DTRAIN);
    end
end
