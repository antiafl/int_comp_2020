clear all
%1er paso, cargar los datos
%path ('C:\Users\veron\OneDrive\Documentos\GitHub\int_comp_2020',path);
path('E:\MASTER\CUATRI_2\2.4_Inteligencia_Computacional\Practicas\int_comp_2020',path);

%Randomiza la semilla
rng('shuffle');

%% IRIS
%load('C:\Users\veron\OneDrive\Documentos\GitHub\int_comp_2020\irisVars');
% load('E:\MASTER\CUATRI_2\2.4_Inteligencia_Computacional\Practicas\int_comp_2020\irisVars');
% loaded = 'iris'; classNumber = 3;

%% CANCER WISONSIN
%load('C:\Users\veron\OneDrive\Documentos\GitHub\int_comp_2020\cancerVars');
load('E:\MASTER\CUATRI_2\2.4_Inteligencia_Computacional\Practicas\int_comp_2020\cancerVars');
loaded = 'cancer';classNumber = 2;

%% PR�CTICA
if (strcmp(loaded,'iris') == 1)
    fprintf('**************************Iris DATASET**************************\n');
%% Entrenamiento   
%Training con Leave 1 Out
    CV = cvpartition(OUTPUTS,'LeaveOut');
    fprintf('1) Entrenando modelos con Leave 1 Out para dataset Iris\n');
    fprintf('\t1.- Discriminante Lineal\n')
    [Mdl_linear] = train_leave1out_discr(CV.NumTestSets,INPUTS,OUTPUTS,'linear',CV);
    
    fprintf('\t2.- Discriminante Cuadr�tico\n')
    [Mdl_quadratic] = train_leave1out_discr(CV.NumTestSets,INPUTS,OUTPUTS,'quadratic',CV);

    fprintf('\t3.- �rboles de Decisi�n\n')
    [Mdl_tree] = train_leave1out_tree(CV.NumTestSets,INPUTS,OUTPUTS,CV,'',0);
    
elseif (strcmp(loaded,'cancer') == 1)
    fprintf('**********Breast Cancer Wisconsin Original DATASET**************\n');
%% Entrenamiento
%Training con K-fold
    k = 10; 
    CV = cvpartition(OUTPUTS,'Kfold',k);
    fprintf('1) Entrenando modelos con K-fold = 10 para dataset Breast Cancer Wisconsin\n');
    fprintf('\t1.- Discriminante Lineal\n')
    [Mdl_linear] = train_Kfold_discr(k,INPUTS,OUTPUTS,'linear',CV);

    fprintf('\t2.- Discriminante Cuadr�tico\n')
    [Mdl_quadratic] = train_Kfold_discr(k,INPUTS,OUTPUTS,'quadratic',CV);

%TODO entrenar 3 �rboles diferentes con par�metros diferentes para el test 
%comprobar que son �rboles distintos
    fprintf('\t3.- �rboles de Decisi�n\n')
    Name = ''; % 'MaxNumSplits'     'MinLeafSize'  'MinParentSize' 'MergeLeaves'
    Value = 0; % (num_ejemplos-1)   (1)            (10)            (on)
    [Mdl_tree] = train_Kfold_tree(k,INPUTS,OUTPUTS,CV,Name,Value);
end
Models = [Mdl_linear; Mdl_quadratic; Mdl_tree];

%% M�tricas de Rendimiento e Informe
%Obtenci�n m�tricas de rendimiento
fprintf('2) Obteniendo m�tricas de rendimiento para los modelos entrenados\n');
%El c�digo se encuentra dentro del bucle por no repetir c�digo innecesario

%TODO
%Sacar m�tricas tambi�n para el entrenamiento y comparar, meter metricas
%entrenamiento en array para despues comparar
%Muestra de las m�tricas obtenidas a trav�s del informe
fprintf('3) INFORME\n') 
if (strcmp(loaded,'iris') == 1) 
    for i=1:size(Models)
        switch i
            case 1
                modelo = 'Discriminante Lineal';
            case 2
                modelo = 'Discriminante Cuadr�tico';
            case 3
                modelo = '�rbol de Decisi�n';
            otherwise
                modelo = '';
        end          
        fprintf('3.%i.1) Informe con las m�tricas para los modelos entrenados con %s\n',i,modelo);
        [mean_ACC(i,:)] = indices_informe_iris(CV.N,classNumber,INPUTS,OUTPUTS,CV,Models(i,:),i);
    end
elseif (strcmp(loaded,'cancer') == 1) 
    for i=1:size(Models)
        switch i
            case 1
                modelo = 'Discriminante Lineal';
            case 2
                modelo = 'Discriminante Cuadr�tico';
            case 3
                modelo = '�rbol de Decisi�n';
            otherwise
                modelo = '';
        end          
        fprintf('3.%i.1) Informe con las m�tricas para los modelos entrenados con %s\n',i,modelo);
        [mean_ACC(i,:)] = indices_informe_cancer(k,1,INPUTS,OUTPUTS,CV,Models(i,:),i);  
    end
end
    
%% Test Estad�stico
fprintf('4) TEST ESTAD�STICO\n');
% Matriz para test estad�stico
matrixForStats =  mean_ACC';

% Matriz con las medias de ACC para cada tipo de modelo
save('matrix4Stats','matrixForStats');

% Test Estad�stico
etiqueta=['Linear          ';'Cuadratico      ';'Arboles Decision'];
[P] = testEstadistico(matrixForStats, etiqueta);