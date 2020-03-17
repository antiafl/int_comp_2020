function [mean_ACC] = indices_informe_iris(N,classNumber,INPUTS,INPUTTRAIN,OUTPUTS,DTRAIN,CV,Mdl,i)

k=1; %controlar que agrupamos los datos de 15 en 15
h=1; %contador para insertar correctamente en la matriz que se le pasa a performance_indexes
for x = 1:N    
    teIdx = CV.test(x);    
    INPUTTEST=INPUTS(teIdx,:);
    DTEST=OUTPUTS(teIdx,:);
    DTEST_REAL=predict(Mdl{x}, INPUTTEST);
    DTEST_array{k,1} = DTEST{1,1};
    DTEST_REAL_array{k,1} = DTEST_REAL{1,1};
    
    %Calcular métricas rendimiento TRAINING
    DTRAIN_REAL=predict(Mdl{x}, INPUTTRAIN{1,x});    
    [CTrain, orderCMTrain] = confusionmat(DTRAIN{1,x}, DTRAIN_REAL);
    for j = 1:classNumber 
        [SensTrainAux(j,k),SpecTrainAux(j,k),PPVTrainAux(j,k),NPVTrainAux(j,k),ACCTrainAux(j,k)] = performance_indexes(CTrain,j);
    end
    %Take elements 15 by 15 to extract metrics
    if (k==15)
        [C, orderCM] = confusionmat(DTEST_array, DTEST_REAL_array);
        for j = 1:classNumber
            [Sens(j,h),Spec(j,h),PPV(j,h),NPV(j,h),ACC(j,h)] = performance_indexes(C,j);
        end  
        SensTrain(:,h) = mean(SensTrainAux, 2);
        SpecTrain(:,h) = mean(SpecTrainAux, 2);
        PPVTrain(:,h) = mean(PPVTrainAux, 2);
        NPVTrain(:,h) = mean(NPVTrainAux, 2);
        ACCTrain(:,h) = mean(ACCTrainAux, 2);
        h=h+1;
        k=1;
    else
        k=k+1;
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
    fprintf('-----IRIS %s-----\n',class)        
    fprintf('\t-TRAIN Accuracy = %3.2f\n', mean(ACCTrain(j,:)))
    fprintf('\t-TEST Accuracy = %3.2f\n\n', mean(ACC(j,:)))
    
    fprintf('\t-TRAIN Recall (Sensibilidad) = %3.2f\n', mean(SensTrain(j,:)))
    fprintf('\t-TEST Recall (Sensibilidad) = %3.2f\n\n', mean(Sens(j,:)))
    
    fprintf('\t-TRAIN Precision = %3.2f\n', mean(PPVTrain(j,:)))
    fprintf('\t-TEST Precision = %3.2f\n\n', mean(PPV(j,:)))
    
    fprintf('\t-TRAIN VPN = %3.2f\n', mean(NPVTrain(j,:)))
    fprintf('\t-TEST VPN = %3.2f\n\n', mean(NPV(j,:)))
    
    fprintf('\t-TRAIN Especificidad = %3.2f\n', mean(SpecTrain(j,:)))
    fprintf('\t-TEST Especificidad = %3.2f\n\n', mean(Spec(j,:)))
    fprintf('\n')
end

fprintf('3.%i.2) Métricas globales\n',i)
fprintf('\t-TRAIN Precisión global (ACCURACY) = %3.2f\n', mean(mean(ACCTrain)));
fprintf('\t-TEST Precisión global (ACCURACY) = %3.2f\n\n', mean(mean(ACC)));

fprintf('\t-TRAIN F1 Score global (F1 SCORE) = %3.2f\n', 2*(mean(mean(PPVTrain))*mean(mean(SensTrain))/(mean(mean(PPVTrain))+mean(mean(SensTrain)))));
fprintf('\t-TEST F1 Score global (F1 SCORE) = %3.2f\n', 2*(mean(mean(PPV))*mean(mean(Sens))/(mean(mean(PPV))+mean(mean(Sens)))));
fprintf('\nFIN INFORME\n\n\n')

mean_ACC = mean(ACC,1);