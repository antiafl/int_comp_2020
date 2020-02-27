%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [P] = testEstadistico(Muestras, etiqueta, criticalValue)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Realiza la comprobacion de si la diferencia entre los minimos de varios
% algoritmos son significativas
% ENTRADAS: 
%  Muestras: matriz SxM, donde en las filas est?n los S valores de la medida de rendimiento (ACC, ECM, etc.) elegida para 
%            comparar los M modelos. Debe haber, al menos, 4 medidas por
%            Modelo.
%  etiqueta: vector que contiene las etiquetas que se utilizaran para
%            identificar a los modelos de acuerdo al orden en que se encuentran en el
%            array de Muestras. Todas las etiquetas deben tener la misma longitud. Por ejemplo, etiqueta=['Disc lineal';'Disc cuadrat';'Arbol Dec   '];
%  criticalValue: valor critico que debe superar el estadistico P para determinar si las muestras de cada Modelo son o no estadisticamente
%           similares. Opcional, valor por defecto 0.05
%
% SALIDAS:  
%   P:       resultado del test,  un p-valor pequeno nos lleva a rechazar la hipotesis nula 
%            y nos permite afirmar que los modelos son distintos
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[S,M]=size(Muestras);

if ~exist('criticalValue')
    criticalValue=0.05
end

% test de Lillie para verificar la normalidad de la muestra. El criticalValue esta limitado a [0.01,0.20] 
% 

H_kt=[];
for c=1:M    
    H_kt=[H_kt,lillietest(mapstd(Muestras(:,c)),criticalValue)];
end

% Si H_kt(i)=1 entonces la hipotesis nula "distribucion normal" se puede
% rechazar con un nivel criticalValue%
    
if sum(H_kt)==0 %si cumple la condicion de normalidad
    %Por defecto los siguientes test devuelven dos figuras: la tabla ANOVA y los box plots de los datos de cada modelo comparado.
    %Si no se quiere mostrar las figuras, hay que anhadir el parametro 'off'
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