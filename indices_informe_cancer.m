function [mean_ACC] = indices_informe_cancer(k,positiveclass,INPUTS,INPUTTRAIN,OUTPUTS,DTRAIN,CV,Mdl,i)

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
fprintf('------------Accuracy------------\n')
fprintf('\t-TRAIN Mean Accuracy = %3.2f\n', mean(ACCTrain))
fprintf('\t-TRAIN Standard Deviation Accuracy = %3.2f\n\n', std(ACCTrain))
fprintf('\t-TEST Mean Accuracy = %3.2f\n', mean(ACC))
fprintf('\t-TEST Standard Deviation Accuracy = %3.2f\n\n', std(ACC))
fprintf('------------Recall------------\n')
fprintf('\t-TRAIN Mean Recall (Sensibilidad) = %3.2f\n', mean(SensTrain))
fprintf('\t-TRAIN Standard Deviation Recall (Sensibilidad) = %3.2f\n\n', std(SensTrain))
fprintf('\t-TEST Mean Recall (Sensibilidad) = %3.2f\n', mean(Sens))
fprintf('\t-TEST Standard Deviation Recall (Sensibilidad) = %3.2f\n\n', std(Sens))
fprintf('------------Precission------------\n')
fprintf('\t-TRAIN Mean Precission = %3.2f\n', mean(PPVTrain))
fprintf('\t-TRAIN Standard Deviation Precission = %3.2f\n\n', std(PPVTrain))
fprintf('\t-TEST Mean Precission = %3.2f\n', mean(PPV))
fprintf('\t-TEST Standard Deviation Precission = %3.2f\n\n', std(PPV))
fprintf('------------VPN------------\n')
fprintf('\t-TRAIN Mean VPN = %3.2f\n', mean(NPVTrain))
fprintf('\t-TRAIN Standard Deviation VPN = %3.2f\n\n', std(NPVTrain))
fprintf('\t-TEST Mean VPN = %3.2f\n', mean(NPV))
fprintf('\t-TEST Standard Deviation VPN = %3.2f\n\n', std(NPV))
fprintf('------------Specificity------------\n')
fprintf('\t-TRAIN Mean Specificity = %3.2f\n', mean(SpecTrain))
fprintf('\t-TRAIN Standard Deviation Specificity = %3.2f\n\n', std(SpecTrain))
fprintf('\t-TEST Mean Specificity = %3.2f\n', mean(Spec))
fprintf('\t-TEST Standard Deviation Specificity = %3.2f\n\n', std(Spec))
fprintf('\n')

fprintf('3.%i.2) Métricas globales\n',i)
fprintf('\t-TRAIN Mean Accuracy (Precisión global) = %3.2f\n', mean(ACCTrain));
fprintf('\t-TRAIN Standard Deviation Accuracy (Precisión global) = %3.2f\n', std(ACCTrain));
fprintf('\t-TEST Mean Accuracy (Precisión global) = %3.2f\n', mean(ACC));
fprintf('\t-TEST Standard Deviation Accuracy (Precisión global) = %3.2f\n\n', std(ACC));

fprintf('\t-TRAIN Mean F1 Score global (F1 SCORE) = %3.2f\n', 2*(mean(PPVTrain)*mean(SensTrain)/(mean(PPVTrain)+mean(SensTrain))));
fprintf('\t-TEST Mean F1 Score global (F1 SCORE) = %3.2f\n\n', 2*(mean(PPV)*mean(Sens)/(mean(PPV)+mean(Sens))));
fprintf('\nFIN INFORME\n\n\n')

mean_ACC = ACC;