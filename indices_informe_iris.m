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
            if (isequal(size(C),[2,2]))
                C=[C ;[0, 0]];
                C=[C [0;0;0]];
            end
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
    fprintf('------------Accuracy------------\n')
    fprintf('\t-TRAIN Mean Accuracy = %3.2f\n', mean(ACCTrain(j,:)))
    fprintf('\t-TRAIN Standard Deviation Accuracy = %3.2f\n\n', std(ACCTrain(j,:)))
    fprintf('\t-TEST Mean Accuracy = %3.2f\n', mean(ACC(j,:)))
    fprintf('\t-TEST Standard Deviation Accuracy = %3.2f\n\n', std(ACC(j,:)))
    fprintf('------------Recall------------\n')
    fprintf('\t-TRAIN Mean Recall (Sensibilidad) = %3.2f\n', mean(SensTrain(j,:)))
    fprintf('\t-TRAIN Standard Deviation Recall (Sensibilidad) = %3.2f\n\n', std(SensTrain(j,:)))
    fprintf('\t-TEST Mean Recall (Sensibilidad) = %3.2f\n', mean(Sens(j,:)))
    fprintf('\t-TEST Standard Deviation Recall (Sensibilidad) = %3.2f\n\n', std(Sens(j,:)))
    fprintf('------------Precission------------\n')
    fprintf('\t-TRAIN Mean Precission = %3.2f\n', mean(PPVTrain(j,:)))
    fprintf('\t-TRAIN Standard Deviation Precission = %3.2f\n\n', std(PPVTrain(j,:)))
    fprintf('\t-TEST Mean Precission = %3.2f\n', mean(PPV(j,:)))
    fprintf('\t-TEST Standard Deviation Precission = %3.2f\n\n', std(PPV(j,:)))
    fprintf('------------VPN------------\n')
    fprintf('\t-TRAIN Mean VPN = %3.2f\n', mean(NPVTrain(j,:)))
    fprintf('\t-TRAIN Standard Deviation VPN = %3.2f\n\n', std(NPVTrain(j,:)))
    fprintf('\t-TEST Mean VPN = %3.2f\n', mean(NPV(j,:)))
    fprintf('\t-TEST Standard Deviation VPN = %3.2f\n\n', std(NPV(j,:)))
    fprintf('------------Specificity------------\n')
    fprintf('\t-TRAIN Mean Specificity = %3.2f\n', mean(SpecTrain(j,:)))
    fprintf('\t-TRAIN Standard Deviation Specificity = %3.2f\n\n', std(SpecTrain(j,:)))
    fprintf('\t-TEST Mean Specificity = %3.2f\n', mean(Spec(j,:)))
    fprintf('\t-TEST Standard Deviation Specificity = %3.2f\n\n', std(Spec(j,:)))
    fprintf('\n')
end

fprintf('3.%i.2) Métricas globales\n',i)
fprintf('\t-TRAIN Mean Accuracy (Precisión global) = %3.2f\n', mean(mean(ACCTrain)));
fprintf('\t-TRAIN Standard Deviation Accuracy (Precisión global) = %3.2f\n', mean(std(ACCTrain)));
fprintf('\t-TEST Mean Accuracy (Precisión global) = %3.2f\n', mean(mean(ACC)));
fprintf('\t-TEST Standard Deviation Accuracy (Precisión global) = %3.2f\n\n', std(std(ACC)));

fprintf('\t-TRAIN Mean F1 Score global (F1 SCORE) = %3.2f\n', 2*(mean(mean(PPVTrain))*mean(mean(SensTrain))/(mean(mean(PPVTrain))+mean(mean(SensTrain)))));
fprintf('\t-TEST Mean F1 Score global (F1 SCORE) = %3.2f\n\n', 2*(mean(mean(PPV))*mean(mean(Sens))/(mean(mean(PPV))+mean(mean(Sens)))));
fprintf('\nFIN INFORME\n\n\n')

mean_ACC = mean(ACC,1);