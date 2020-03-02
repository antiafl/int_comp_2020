function [] = informe(j,ACC,Sens,PPV,NPV,Spec)

fprintf('Accuracy = %3.2f\n', mean(ACC(j,:)))
fprintf('Recall (Sensibilidad) = %3.2f\n', mean(Sens(j,:)))
fprintf('Precision = %3.2f\n', mean(PPV(j,:)))
fprintf('VPN = %3.2f\n', mean(NPV(j,:)))
fprintf('Especificidad = %3.2f\n', mean(Spec(j,:)))
fprintf('\n')