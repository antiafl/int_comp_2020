function [mean_ACC] = indices_informe_cancer(k,positiveclass,INPUTS,OUTPUTS,CV,Mdl)

for i = 1:k    
    teIdx = CV.test(i);    
    INPUTTEST=INPUTS(teIdx,:);
    DTEST=OUTPUTS(teIdx,:);
    DTEST_REAL=predict(Mdl{i}, INPUTTEST);
    [C, orderCM] = confusionmat(DTEST, DTEST_REAL);
    [Sens(i),Spec(i),PPV(i),NPV(i),ACC(i)] = performance_indexes(C,positiveclass);
end
fprintf('Para datos de TEST de Cáncer Wisconsin\n');
fprintf('Accuracy = %3.2f\n', mean(ACC))
fprintf('Recall (Sensibilidad) = %3.2f\n', mean(Sens))
fprintf('Precision = %3.2f\n', mean(PPV))
fprintf('VPN = %3.2f\n', mean(NPV))
fprintf('Especificidad = %3.2f\n', mean(Spec))
fprintf('\n')

fprintf('4) Métricas globales\n')
fprintf('Precisión global (ACCURACY) = %3.2f\n', mean(ACC));
fprintf('F1 Score global (F1 SCORE) = %3.2f\n', 2*(mean(PPV)*mean(Sens)/(mean(PPV)+mean(Sens))));
fprintf('\nFIN INFORME\n')

mean_ACC = ACC;