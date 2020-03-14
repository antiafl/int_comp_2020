function [mean_ACC] = indices_informe_cancer(k,positiveclass,INPUTS,INPUTTRAIN,OUTPUTS,DTRAIN,CV,Mdl,i)

%TODO - pensar si vale la pena guardar C matriz de confusión para después
for x = 1:k    
    teIdx = CV.test(x);    
    INPUTTEST=INPUTS(teIdx,:);
    DTEST=OUTPUTS(teIdx,:);
    DTEST_REAL=predict(Mdl{x}, INPUTTEST);
    DTRAIN_REAL=predict(Mdl{x}, INPUTTRAIN{x});
    [C, orderCM] = confusionmat(DTEST, DTEST_REAL);
    [CTrain, orderCMtrain] = confusionmat(DTRAIN{x}, DTRAIN_REAL);
    [Sens(x),Spec(x),PPV(x),NPV(x),ACC(x)] = performance_indexes(C,positiveclass);
    [SensTrain(x),SpecTrain(x),PPVTrain(x),NPVTrain(x),ACCTrain(x)] = performance_indexes(CTrain,positiveclass);
end
fprintf('\t-TRAIN Accuracy = %3.2f\n', mean(ACCTrain))
fprintf('\t-TEST Accuracy = %3.2f\n\n', mean(ACC))

fprintf('\t-TRAIN Recall (Sensibilidad) = %3.2f\n', mean(SensTrain))
fprintf('\t-TEST Recall (Sensibilidad) = %3.2f\n\n', mean(Sens))

fprintf('\t-TRAIN Precision = %3.2f\n', mean(PPVTrain))
fprintf('\t-TEST Precision = %3.2f\n\n', mean(PPV))

fprintf('\t-TRAIN VPN = %3.2f\n', mean(NPVTrain))
fprintf('\t-TEST VPN = %3.2f\n\n', mean(NPV))

fprintf('\t-TRAIN Especificidad = %3.2f\n', mean(SpecTrain))
fprintf('\t-TEST Especificidad = %3.2f\n\n', mean(Spec))
fprintf('\n')

fprintf('3.%i.2) Métricas globales\n',i)
fprintf('\t-TRAIN Precisión global (ACCURACY) = %3.2f\n', mean(ACCTrain));
fprintf('\t-TEST Precisión global (ACCURACY) = %3.2f\n\n', mean(ACC));

fprintf('\t-TRAIN F1 Score global (F1 SCORE) = %3.2f\n', 2*(mean(PPVTrain)*mean(SensTrain)/(mean(PPVTrain)+mean(SensTrain))));
fprintf('\t-TEST F1 Score global (F1 SCORE) = %3.2f\n\n', 2*(mean(PPV)*mean(Sens)/(mean(PPV)+mean(Sens))));
fprintf('\nFIN INFORME\n\n\n')

mean_ACC = ACC;