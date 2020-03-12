function [model] = train_leave1out_tree(N,INPUTS,OUTPUTS,CV,splitCriterion,varargin)
predictorNames = {'SepalLength','SepalWidth','PetalLength','PetalWidth'};
responseName = 'Iris_type';
% 'gdi' 'twoing'
for i = 1:N
    trIdx = CV.training(i);   
    INPUTTRAIN=INPUTS(trIdx,:);
    DTRAIN=OUTPUTS(trIdx,:); 
    model{i} = fitctree(INPUTTRAIN, DTRAIN,'PredictorNames', predictorNames,'ResponseName',responseName, 'SplitCriterion', splitCriterion);
end
