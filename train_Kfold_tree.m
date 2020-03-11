function [model] = train_Kfold_tree(k,INPUTS,OUTPUTS,CV,Name,Value)

PredictorNames = {'Clump_Thickness', 'Uniformity_Size', 'Uniformity_Shape', 'Marginal_Adhesion', 'Single_Epithelial_Size', 'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses'};
ResponseName = 'Tumor_Type';
SplitCriterion = 'twoing';
%'gdi' 'deviance'

for i = 1:k
    trIdx = CV.training(i);   
    INPUTTRAIN=INPUTS(trIdx,:);
    DTRAIN=OUTPUTS(trIdx,:);
    if (~strcmp(Name,'')==1) && (Value~=0)
        model{i} = fitctree(INPUTTRAIN, DTRAIN, Name, Value, 'PredictorNames',PredictorNames, 'ResponseName',ResponseName, 'SplitCriterion', SplitCriterion);
    else
        model{i} = fitctree(INPUTTRAIN, DTRAIN, 'PredictorNames',PredictorNames, 'ResponseName',ResponseName,'SplitCriterion', SplitCriterion);
    end
end

        
