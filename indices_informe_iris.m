function [mean_ACC] = indices_informe_iris(N,classNumber,INPUTS,OUTPUTS,CV,Mdl,i)

%TODO - pensar si vale la pena guardar C matriz de confusión para después
k=1;
h=1;
for x = 1:N    
    teIdx = CV.test(x);    
    INPUTTEST=INPUTS(teIdx,:);
    DTEST=OUTPUTS(teIdx,:);
    DTEST_REAL=predict(Mdl{x}, INPUTTEST);
    DTEST_array{k,1} = DTEST{1,1};
    DTEST_REAL_array{k,1} = DTEST_REAL{1,1};
    %Take elements 15 by 15 to extract metrics
    if (k==15)
        [C, orderCM] = confusionmat(DTEST_array, DTEST_REAL_array);
        for j = 1:classNumber
            [Sens(j,h),Spec(j,h),PPV(j,h),NPV(j,h),ACC(j,h)] = performance_indexes(C,j);
        end
        h=h+1;
        k=1;
    else
        k=k+1;
        continue
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
    fprintf('\tIRIS %s \n',class)        
    fprintf('\t-Accuracy = %3.2f\n', mean(ACC(j,:)))
    fprintf('\t-Recall (Sensibilidad) = %3.2f\n', mean(Sens(j,:)))
    fprintf('\t-Precision = %3.2f\n', mean(PPV(j,:)))
    fprintf('\t-VPN = %3.2f\n', mean(NPV(j,:)))
    fprintf('\t-Especificidad = %3.2f\n', mean(Spec(j,:)))
    fprintf('\n')
end

fprintf('3.%i.2) Métricas globales\n',i)
fprintf('\t-Precisión global (ACCURACY) = %3.2f\n', mean(mean(ACC)));
fprintf('\t-F1 Score global (F1 SCORE) = %3.2f\n', 2*(mean(mean(PPV))*mean(mean(Sens))/(mean(mean(PPV))+mean(mean(Sens)))));
fprintf('\nFIN INFORME\n\n\n')

mean_ACC = mean(ACC,1);