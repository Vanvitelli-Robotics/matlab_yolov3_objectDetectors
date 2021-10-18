%script per generare i grafici recall x precision
clc, clear all, close all
%si caricano i risultati del testing, in particolare vengono usati gli
%array recall e precision restituiti dalla funzione
%evaluateDetectionPrecision. Viene anche caricata una tabella il cui solo
%scopo è fornire i nomi delle classi
load('risultati.mat');
load('finishTbl.mat');
for i=1:6
    figure
    plot(recall{i,1},precision{i,1})
    xlabel('Recall')
    ylabel('Precision')
    grid on
    title(sprintf('Average Precision = %.4f, class %s', ap(i),baleaTbl.Properties.VariableNames{1,i+1}),'Interpreter','none')
end