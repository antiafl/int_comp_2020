function [model,INPUTTRAIN,DTRAIN] = train_Kfold_tree(k,INPUTS,OUTPUTS,CV,splitCriterion,varargin)

predictorNames = {'Clump_Thickness', 'Uniformity_Size', 'Uniformity_Shape', 'Marginal_Adhesion', 'Single_Epithelial_Size', 'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses'};
responseName = 'Tumor_Type';
for i = 1:k
    trIdx = CV.training(i);   
    INPUTTRAIN{i}=INPUTS(trIdx,:);
    DTRAIN{i}=OUTPUTS(trIdx,:);
    model{i} = fitctree(INPUTTRAIN{i}, DTRAIN{i}, 'PredictorNames',predictorNames, 'ResponseName',responseName, 'SplitCriterion', splitCriterion);
end

        
