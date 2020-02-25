clear all
%1er paso, cargar los datos
% path('E:\MASTER\CUATRI_2\2.4_Inteligencia_Computacional\Practicas\p1',path);
path ('C:\Users\veron\OneDrive\Documentos\GitHub\int_comp_2020',path);

% load('E:\MASTER\CUATRI_2\2.4_Inteligencia_Computacional\Practicas\p0\irisVars');
load('C:\Users\veron\OneDrive\Documentos\GitHub\int_comp_2020\irisVars');

%Training con K-fold
k = 10; 
type = 'quadratic';
fprintf('1) Entrenando modelos con K-fold = 10 y usando %s discriminant\n',type);
CV = cvpartition(OUTPUTS,'Kfold',k);
[Mdl] = train_linearDisc_Kfold_p1(k,INPUTS,OUTPUTS,type,CV);

%Obtención métricas de rendimiento
fprintf('2) Obteniendo métricas de rendimiento para los modelos entrenados\n');
classNumber = 3;
[Sens,Spec,PPV,NPV,ACC] = indices_Kfold_p1(k,classNumber,INPUTS,OUTPUTS,CV,Mdl);

fprintf('3)INFORME\n') 
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
fprintf('4) Métricas globales\n')
fprintf('Precisión global (ACCURACY) = %3.2f\n', mean(mean(ACC)));
fprintf('F1 Score global (F1 SCORE) = %3.2f\n', 2*(mean(mean(PPV))*mean(mean(Sens))/(mean(mean(PPV))+mean(mean(Sens)))));
fprintf('\nFIN INFORME\n')

% Carga de la matriz con las sensibiliades de los modelos para realizar el test estadístico
if exist('matrix4Stats.mat', 'file') == 2
     load('matrix4Stats')
end

% matrixForStats=zeros(4,10);

% Cálculo de la matriz para test estadístico
switch type
    case 'linear'
        for i = 1:k
            matrixForStats(1,i) =  mean(ACC(:,i));
        end
    case 'quadratic'
        for i = 1:k
            matrixForStats(2,i) =  mean(ACC(:,i));
        end
end

save('matrix4Stats','matrixForStats');

% Test Estadístico
etiqueta=['Linear  ','Cuadratico  ','Prueba1  ','Prueba2  '];
% [P] = testEstadistico(matrixForStats, etiqueta)