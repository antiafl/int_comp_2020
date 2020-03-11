function [model] = train_leave1out_tree(N,INPUTS,OUTPUTS,CV,Name,Value)
predictorNames = {'SepalLength','SepalWidth','PetalLength','PetalWidth'};
responseName = 'Iris_type';
splitCriterion = 'deviance';
% 'gdi' 'twoing'
for i = 1:N
    trIdx = CV.training(i);   
    INPUTTRAIN=INPUTS(trIdx,:);
    DTRAIN=OUTPUTS(trIdx,:); 
    if (~strcmp(Name,'')==1) && (Value~=0)
        model{i} = fitctree(INPUTTRAIN, DTRAIN, Name, Value, 'PredictorNames', predictorNames,'ResponseName',responseName, 'SplitCriterion', 'deviance');
    else
        model{i} = fitctree(INPUTTRAIN, DTRAIN,'PredictorNames', predictorNames,'ResponseName',responseName, 'SplitCriterion', 'deviance');
    end
end
