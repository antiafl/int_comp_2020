function [model,INPUTTRAIN,DTRAIN] = train_leave1out_tree(N,INPUTS,OUTPUTS,CV,splitCriterion,varargin)

predictorNames = {'SepalLength','SepalWidth','PetalLength','PetalWidth'};
responseName = 'Iris_type';
for i = 1:N
    trIdx = CV.training(i);   
    INPUTTRAIN{i}=INPUTS(trIdx,:);
    DTRAIN{i}=OUTPUTS(trIdx,:); 
    model{i} = fitctree(INPUTTRAIN{i}, DTRAIN{i},'PredictorNames', predictorNames,'ResponseName',responseName, 'SplitCriterion', splitCriterion);
end
