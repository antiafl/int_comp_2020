function [mean_ACC] = indices_informe_cancer(k,positiveclass,INPUTS,OUTPUTS,CV,Mdl,i)

%TODO - pensar si vale la pena guardar C matriz de confusión para después
for x = 1:k    
    teIdx = CV.test(x);    
    INPUTTEST=INPUTS(teIdx,:);
    DTEST=OUTPUTS(teIdx,:);
    DTEST_REAL=predict(Mdl{x}, INPUTTEST);
    [C, orderCM] = confusionmat(DTEST, DTEST_REAL);
    [Sens(x),Spec(x),PPV(x),NPV(x),ACC(x)] = performance_indexes(C,positiveclass);
end
fprintf('\t-Accuracy = %3.2f\n', mean(ACC))
fprintf('\t-Recall (Sensibilidad) = %3.2f\n', mean(Sens))
fprintf('\t-Precision = %3.2f\n', mean(PPV))
fprintf('\t-VPN = %3.2f\n', mean(NPV))
fprintf('\t-Especificidad = %3.2f\n', mean(Spec))
fprintf('\n')

fprintf('3.%i.2) Métricas globales\n',i)
fprintf('\t-Precisión global (ACCURACY) = %3.2f\n', mean(ACC));
fprintf('\t-F1 Score global (F1 SCORE) = %3.2f\n', 2*(mean(PPV)*mean(Sens)/(mean(PPV)+mean(Sens))));
fprintf('\nFIN INFORME\n\n\n')

mean_ACC = ACC;