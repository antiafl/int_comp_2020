function [model] = train_Kfold_tree(k,INPUTS,OUTPUTS,CV,Name,Value)

for i = 1:k
    trIdx = CV.training(i);   
    INPUTTRAIN=INPUTS(trIdx,:);
    DTRAIN=OUTPUTS(trIdx,:);
    if (~strcmp(Name,'')==1) && (Value~=0)
        model{i} = fitctree(INPUTTRAIN, DTRAIN, Name, Value);
    else
        model{i} = fitctree(INPUTTRAIN, DTRAIN);
    end
end

        
