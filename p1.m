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

%% DATASET
if (strcmp(loaded,'iris') == 1)
    fprintf('**************************Iris DATASET**************************\n');
elseif (strcmp(loaded,'cancer') == 1)
    fprintf('**********Breast Cancer Wisconsin Original DATASET**************\n');    
end
%% ENTRENAMIENTO
%Tipos de Modelos
type = {'linear', 'quadratic', 'tree'};

%Training con K-fold
k = 10; 
CV = cvpartition(OUTPUTS,'Kfold',k);
fprintf('1) Entrenando modelos con K-fold = 10\n');

fprintf('\t1.- Discriminante Lineal\n')
[Mdl_linear] = train_Kfold_p1(k,INPUTS,OUTPUTS,type{1},CV);

fprintf('\t2.- Discriminante Cuadr�tico\n')
[Mdl_quadratic] = train_Kfold_p1(k,INPUTS,OUTPUTS,type{2},CV);

fprintf('\t3.- �rboles de Decisi�n\n')
[Mdl_tree] = train_Kfold_p1(k,INPUTS,OUTPUTS,type{3},CV);

Models = [Mdl_linear; Mdl_quadratic; Mdl_tree];
%% M�TRICAS DE RENDIMIENTO E INFORME
%Obtenci�n m�tricas de rendimiento
fprintf('2) Obteniendo m�tricas de rendimiento para los modelos entrenados\n');
%El c�digo se encuentra dentro del bucle por no repetir c�digo innecesario

%Muestra de las m�tricas obtenidas a trav�s del informe
fprintf('3) Informe\n') 
if (strcmp(loaded,'iris') == 1) 
    for i=1:size(type')
        [mean_ACC(i,:)] = indices_informe_iris(k,classNumber,INPUTS,OUTPUTS,CV,Models(i,:));
    end
elseif (strcmp(loaded,'cancer') == 1) 
    for i=1:size(type')
        [mean_ACC(i,:)] = indices_informe_cancer(k,1,INPUTS,OUTPUTS,CV,Models(i,:));  
    end
end
    
%% TEST ESTAD�STICO

% Matriz para test estad�stico
matrixForStats =  mean_ACC';

% Matriz con las medias de ACC para cada tipo de modelo
save('matrix4Stats','matrixForStats');

% Test Estad�stico
etiqueta=['Linear          ';'Cuadratico      ';'Arboles Decision'];
[P] = testEstadistico(matrixForStats, etiqueta);