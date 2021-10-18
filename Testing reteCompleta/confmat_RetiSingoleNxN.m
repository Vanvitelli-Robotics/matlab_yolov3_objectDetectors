 clear all
 close all
load('beckmann_demoTbl');
load ('yoloDenkmitAggiunta.mat');
testTbl=DemoCompletetbl;
imdsTest = imageDatastore(testTbl.imageFilename);
bldsTest = boxLabelDatastore(testTbl(:, 2:end));
testData = combine(imdsTest, bldsTest);
results = detect(yolov3Detector,testData,'MiniBatchSize',16);
fp=0; fn=0; tp=0; ct=0; predictedlabels=[]; truelabels=[];
for i=1:height(results)
    iouarray=[];
    indexarray=[];
    labels=results(i,'Labels');
    labels=labels{1,1};
    labels=labels{1,1};%si ricavano tutte le label predette
    if(~isempty(labels))
        [~,idx]=max(cell2mat(table2cell(results(i,'Scores'))));%si salva l'indice relativo allo score più alto
        
        bboxes=results(i,'Boxes');
        bboxes=bboxes{1,1};
        bboxes=bboxes{1,1};%si salvano tutte le bbox predette
        
        label=labels(idx,1);
        bbox=bboxes(idx,:);%vengono considerate solo la label e la bbox relative allo score massimo
       
        %la base del funzionamento di questo modo di operare nel ciclo for
        %è che ci sia una sola gtruth per ogni oggetto in ogni immagine.
        for j=2:25
            if(~isempty(cell2mat(table2cell(testTbl(i,j)))))
                iouarray=[iouarray bb_intersection_over_union(uint16(bbox),uint16(cell2mat(table2cell(testTbl(i,j)))))];%si salva in un array le iou della bbox predetta con tutte quelle presenti
                indexarray=[indexarray j];
            end
        end
        %utilizando gli array costruiti nel ciclo for precedente si
        %verifica sia che l'iou sia > 0,5 sia che l'etichetta è corretta.
        %é a questo punto che si distinguono i casi di tp,fp e fn
        [iou, iouindex]=max(iouarray);
        index=indexarray(iouindex);
        predictedlabels=[predictedlabels label];
        truelabels=[truelabels testTbl.Properties.VariableNames(index)];
        if(iou>0.5 && strcmp(cellstr(label),testTbl.Properties.VariableNames(index)))
            tp=tp+1;
        else
            fp=fp+1;
        end
    else
        fn=fn+1;      %falso negativo, non si è rilevato alcun oggetto, ma nelle immagini
                      %prese in esame è sempre presente un oggetto
    end
end
A=[tp fp;fn 0];
% Precision=tp/(tp+fp);
% Recall=tp/(tp+fn);
% Fscore=2*((Precision * Recall)/(Precision + Recall));
figure
M=confusionchart(truelabels,cellstr(predictedlabels),'RowSummary','row-normalized','ColumnSummary','column-normalized');
confusionmat(truelabels,cellstr(predictedlabels));