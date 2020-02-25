%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [P] = testEstadistico(Muestras, etiqueta, criticalValue)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Realiza la comprobacion de si la diferencia entre los minimos de varios
% algoritmos son significativas
% ENTRADAS: Muestras: matriz SxM que contiene los S valores para la medida de rendimiento (precisión, ECM, etc.) elegida para 
%           caracterizar los M modelos a comparar. 
%           etiqueta: vector que contiene las etiquetas que se utilizaran
%           para identificar a los modelos de acuerdo al orden en que se encuentran en el cell array
%           Muestras. Por ejemplo, etiqueta=['SCG  ';'GDX  ';'miGDX'];
%           criticalValue: valor critico a aplicar que debe superar el estadistico P
%           para determinar si las muestras son o no estadisticamente
%           similares. Opcional, valor por defecto 0.05
% SALIDAS:  P: resultado del test,  un p-valor pequeño nos llevaría a rechazar 
%           la hipótesis nula y nos haría afirmar que los modelos son distintas
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[S,M]=size(Muestras);
%Vector de minimos debe contener en cada columna los minimos de un
%algoritmo

if ~exist('criticalValue')
    criticalValue=0.05
end

% test para verificar la normalidad de la muestra. Mas fiable que kstest el criticalValue esta limitado a [0.01,0.20] 
% Debe haber al menos 4 medidas por Modelo

H_kt=[];
for c=1:M    
    H_kt=[H_kt,lillietest(mapstd(Muestras(:,c)),criticalValue)];
end

% Si H_kt(i)=1 entonces la hipotesis nula "distribución normal" se puede
% rechazar con un nivel 5%
    
if sum(H_kt)==0 %si cumple la condicion de normalidad
    %Por defecto los siguientes test devuelven dos figuras: la tabla ANOVA y los box plots de los datos de cada modelo comparado.
    %Si no se quiere mostrar las figuras, hay que añadir el parámetro 'off'
    [P,ANOVATAB,STATS]  = anova1(Muestras,etiqueta); %[P,ANOVATAB,STATS]  = anova1(Muestras,etiMqueta,'off');
else
    [P,ANOVATAB,STATS]  = kruskalwallis(Muestras,etiqueta); %[P,ANOVATAB,STATS]  = kruskalwallis(Muestras,etiqueta,'off');
end

if P<0.10
%si la probabilidad de que las muestras sean iguales es menor de 0.10    
%Hacemos un test de comparacion multiple
    %STATS.gnames=etiqueta;
    figure;
    [c,m,h,nms] = multcompare(STATS,'display','on');%,'display','off');
end