clear all
%1er paso, cargar los datos
%path ('C:\Users\veron\OneDrive\Documentos\GitHub\int_comp_2020',path);
path('E:\MASTER\CUATRI_2\2.4_Inteligencia_Computacional\Practicas\int_comp_2020',path);

%Randomiza la semilla
rng('shuffle');

%% IRIS
%load('C:\Users\veron\OneDrive\Documentos\GitHub\int_comp_2020\irisVars');
%load('E:\MASTER\CUATRI_2\2.4_Inteligencia_Computacional\Practicas\int_comp_2020\irisVars');
%loaded = 'iris'; classNumber = 3;

%% CANCER WISONSIN
%load('C:\Users\veron\OneDrive\Documentos\GitHub\int_comp_2020\cancerVars');
load('E:\MASTER\CUATRI_2\2.4_Inteligencia_Computacional\Practicas\int_comp_2020\cancerVars');
loaded = 'cancer';classNumber = 2;

%% ENTRENAMIENTO
%Tipos de Modelos
type = {'linear', 'quadratic', 'tree'};
selected_type = type{1};

%Training con K-fold
k = 10; 
fprintf('1) Entrenando modelos con K-fold = 10');
switch selected_type
    case 'linear'
        fprintf('\tDiscriminante Lineal\n')
    case 'quadratic'
        fprintf('\tDiscriminante Cuadr�tico\n')
    case 'tree'
        fprintf('\t�rboles de Decisi�n\n')
end

CV = cvpartition(OUTPUTS,'Kfold',k);
[Mdl] = train_Kfold_p1(k,INPUTS,OUTPUTS,selected_type,CV);

%% M�TRICAS DE RENDIMIENTO E INFORME
%Obtenci�n m�tricas de rendimiento
fprintf('2) Obteniendo m�tricas de rendimiento para los modelos entrenados\n');
[Sens,Spec,PPV,NPV,ACC] = indices_Kfold_p1(k,classNumber,INPUTS,OUTPUTS,CV,Mdl);

%Muestra de las m�tricas obtenidas a trav�s del informe
fprintf('3) Informe\n') 
if (strcmp(loaded,'iris') == 1)    
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
        informe(j,ACC,Sens,PPV,NPV,Spec);
    end
elseif (strcmp(loaded,'cancer') == 1)
    for j = 1:classNumber
        switch j
            case 1
                class = 'Tumor BENIGNO';
            case 2
                class = 'Tumor MALIGNO';
            otherwise
                class = '';
        end  
        fprintf('Para datos de TEST de C�ncer Wisconsin:  %s \n',class);
        informe(j,ACC,Sens,PPV,NPV,Spec);
    end    
end
    
fprintf('4) M�tricas globales\n')
fprintf('Precisi�n global (ACCURACY) = %3.2f\n', mean(mean(ACC)));
fprintf('F1 Score global (F1 SCORE) = %3.2f\n', 2*(mean(mean(PPV))*mean(mean(Sens))/(mean(mean(PPV))+mean(mean(Sens)))));
fprintf('\nFIN INFORME\n')

%% TEST ESTAD�STICO
% Carga de la matriz con las sensibiliades de los modelos para realizar el test estad�stico
if exist('matrix4Stats.mat', 'file') == 2
     load('matrix4Stats')
end

% C�lculo de la matriz para test estad�stico
switch selected_type
    case 'linear'        
        matrixForStats(1,:) =  mean(ACC,1);
    case 'quadratic'
        matrixForStats(2,:) =  mean(ACC,1);
    case 'tree'
        matrixForStats(3,:) =  mean(ACC,1);
end

% Matriz con las medias de ACC para cada tipo de modelo
save('matrix4Stats','matrixForStats');

% Test Estad�stico
etiqueta=['Linear          ';'Cuadratico      ';'Arboles Decision';'Prueba2         '];
[P] = testEstadistico(matrixForStats', etiqueta);