function [Sens,Spec,PPV,NPV,ACC] = indices_Kfold_p1(k,classNumber,INPUTS,OUTPUTS,CV,Mdl)

for j = 1:classNumber
    for i = 1:k    
        teIdx = CV.test(i);    
        INPUTTEST=INPUTS(teIdx,:);
        DTEST=OUTPUTS(teIdx,:);
        DTEST_REAL=predict(Mdl{i}, INPUTTEST);
        [C, orderCM] = confusionmat(DTEST, DTEST_REAL);
        [Sens(j,i),Spec(j,i),PPV(j,i),NPV(j,i),ACC(j,i)] = performance_indexes(C,j);
    end
end