function [mean_ACC] = indices_informe_iris(k,classNumber,INPUTS,OUTPUTS,CV,Mdl)


for i = 1:k    
    teIdx = CV.test(i);    
    INPUTTEST=INPUTS(teIdx,:);
    DTEST=OUTPUTS(teIdx,:);
    DTEST_REAL=predict(Mdl{i}, INPUTTEST);
    [C, orderCM] = confusionmat(DTEST, DTEST_REAL);
    for j = 1:classNumber
        [Sens(j,i),Spec(j,i),PPV(j,i),NPV(j,i),ACC(j,i)] = performance_indexes(C,j);
    end
end
for j = 1:classNumber
    switch j
        case 1
            class = 'SETOSA';
        case 2
            class = 'VERSICOLOR';
        case 3
            class = 'VIRGINICA';
        otherwise
            class = '';
    end          
    fprintf('Para datos de TEST de IRIS %s \n',class)        
    fprintf('Accuracy = %3.2f\n', mean(ACC(j,:)))
    fprintf('Recall (Sensibilidad) = %3.2f\n', mean(Sens(j,:)))
    fprintf('Precision = %3.2f\n', mean(PPV(j,:)))
    fprintf('VPN = %3.2f\n', mean(NPV(j,:)))
    fprintf('Especificidad = %3.2f\n', mean(Spec(j,:)))
    fprintf('\n')
end

mean_ACC = mean(ACC,1);