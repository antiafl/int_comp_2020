function [Sens,Spec,PPV,NPV,ACC] = performance_indexes(CM,PositiveClass)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculates Performance Indexes for a classification task. 
% It takes the as positive class the one indicated by the user.
% Parameters of the function:
% --------------------------
% Inputs:
%   CM:             Confusion Matrix (CxC, where C is the number of classes. CM(I,J) represents the count of instances of class I and whose predicted class is J
%   PositiveClass:  the index of the positive class (between 1 and C)
% Returns:
%   Sens:       Sensitivity of the classifier.
%   Spec:       Specificity of the classifier.
%   PPV:        Positive Predicted Value
%   NPV:        Negative Predicted Value
%   ACC:        Global accuracy of the classifier.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% transforma la CM en una de dos clases 
classNum=size(CM,1);
%verdaderos positivos
TP=CM(PositiveClass,PositiveClass);
%verdaderos negativos, falsos positivos Y falsos negativos
TN=0;
FP=0;
FN=0;
for REAL=1:classNum
        for PREDICTED=1:classNum
                if (REAL~=PositiveClass)&&(PREDICTED~=PositiveClass) TN=TN+CM(REAL,PREDICTED); end 
                if (REAL~=PositiveClass)&&(PREDICTED==PositiveClass) FP=FP+CM(REAL,PREDICTED); end 
                if (REAL==PositiveClass)&&(PREDICTED~=PositiveClass) FN=FN+CM(REAL,PREDICTED); end
        end
end

%TPR=TP/(TP+FN) o sensibilidad
if (TP+FN)==0
    fprintf('No es posible calcular la tasa de TP al no haber ejemplos positivos\n');
    Sens=NaN;
else
    Sens= TP/(TP+FN);
end

%TNR= TN/(FP+TN) o Especificidad
if (TN+FP)==0
    fprintf('No es posible calcular la Especificidad al no haber ejemplos negativos\n');
    Spec=NaN;
else
    Spec= TN/(TN+FP);
end
%PPV=TP/(TP+FP)
if TP+FP==0
    fprintf('No es posible calcular la tasa el VPP al no haber ejemplos clasificados en la clase positiva\n');
    PPV=NaN;
else
    PPV=TP/(TP+FP);
end

%NPV=TN/(TN+FN)
if TN+FN==0
    fprintf('No es posible calcular la tasa el VPN al no haber ejemplos clasificados en la clase negativa\n');
    NPV=NaN;
else
    NPV=TN/(TN+FN);
end

ACC=(TP+TN)/(TP+TN+FP+FN);

