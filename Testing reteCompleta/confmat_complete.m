clc
clear all
load('DemoCompletetbl.mat');
load ('yoloCompletoFixed.mat');
%load('results.mat');
testTbl=DemoCompletetbl;
imdsTest = imageDatastore(testTbl.imageFilename);
bldsTest = boxLabelDatastore(testTbl(:, 2:end));
testData = combine(imdsTest, bldsTest);
results = detect(yolov3Detector,testData,'MiniBatchSize',16);
fp=0; fn=0; tp=0; predictedlabels=[]; truelabels=[];
totalprediction=0; totalGtruth=0; 
for i=1:height(results)
    
    indiciBBoxDataset=[];
    
    labels=results(i,'Labels');
    labels=labels{1,1};
    labels=labels{1,1};%ricavo tutte le label predette
    
    bboxes=results(i,'Boxes');
        bboxes=bboxes{1,1};
        bboxes=bboxes{1,1};%ricavo tutte le bbox predette
        
    oggetti=["Beckmann";"Balea";"Denkmit";"Finish"];%mi serve per controllare che solo 1 bbox per ogni oggetto venga generata
   
    if(~isempty(labels))
        
        dim=size(labels);
        numLabelsPredicted=dim(1);%ricavo il nuemro di label generate
        %totalprediction=totalprediction+numLabelsPredicted;%calcolo le total prediction della rete
        
     for n=1:4 %scorro tutti gli oggetti ed effettuo il cotrollo
         
        indiciLabel=[];
        indexmax=0; 
        numRipetizioni=0;%indica quante volte viene prodotta la label di uno stesso oggetto
        
        
        for j=1:numLabelsPredicted
              x=strfind(string(labels(j,1)),oggetti(n,1));%ad ogni iterazone del ciclo n, verifico in quali label è presente il nome dell oggetto n-esimo
              if(size(x)>0)%vado qui solo le la label contiene quel nome
                  indiciLabel=[indiciLabel j];%salvo gli indici delle label prese
                  x=[];
              end
        
        end
        numRipetizioni=size(indiciLabel);
        numRipetizioni=numRipetizioni(2);%ricavo il nuemro di indici e quindi di ripetizioni
        
        if(numRipetizioni>1)%ovvero se c'è più di una label riferita allo stesso oggetto
            
            score=cell2mat(results.Scores(i));%salvo gli score delle bbox 
           
            indexmax=0;
            maxscore=0;
            
            for j=1:numRipetizioni
                if(maxscore < score(indiciLabel(1,j),1))
                   indexmax=indiciLabel(1,j);%salvo l'indice dello score massimo
                   maxscore=score(indiciLabel(1,j),1);%agiorno lo score massimo
                end
            end
            
            for j=1:numRipetizioni
                if(~(indiciLabel(1,j)==indexmax))
                labels(indiciLabel(1,j),:)=[];%elimino le label che non corrispondono allo score massimo
                bboxes(indiciLabel(1,j),:)=[];%elimino le bbox che non corrispondono allo score massimo
                score(indiciLabel(1,j))=[];%elimino lo score
                end
            end
            dim=size(labels);
            numLabelsEffective=dim(1);%ricavo il nuemro di label considerate dopo aver eliminato le duplicazioni
            
            results(i,1)=cell2table(mat2cell(bboxes,numLabelsEffective));
            results(i,2)=cell2table(mat2cell(score,numLabelsEffective));
            results(i,3)=cell2table(mat2cell(labels,numLabelsEffective));
            
            %ho aggiornato la tabella dei risultati, togliendo el bbox che non voglio condierare
        end
        
        dim=size(labels);
        numLabelsPredicted=dim(1);%aggiorno il nuemero delle label, non mi inficia il conteggio finale delle predizioni poiché lo faccio prima del ciclo in n
        
     end    
     
      totalprediction=totalprediction+numLabelsPredicted;
        
        for a=2:25 
            if(~isempty(cell2mat(table2cell(testTbl(i,a)))))%controllo se esiste e con che etichetta un oggetto nell'immagine, salvandomi l'indice della coloi-esima
            indiciBBoxDataset=[indiciBBoxDataset a];
            end
        end   
        
        dim=size(indiciBBoxDataset);
        numLabelsPresent=dim(2);%numero di gtruth presenti
        totalGtruth=totalGtruth+numLabelsPresent;
        
        dim=size(labels);
        numLabelsEffective=dim(1);%ricavo il nuemro di label considerate dopo aver eliminato le duplicazioni
        
        for k=1:numLabelsEffective
            label=labels(k,1);
            bbox=bboxes(k,:);
            MaxIou=0;
            indexMaxIou=0;
            for t=1:numLabelsPresent
               bboxT=testTbl(i,indiciBBoxDataset(t)); %box gTruth, mi prendo la bbox che effettivamente ho messo manualmente, la gtruth
               bboxT=cell2mat(table2cell(bboxT));  
               iou=bb_intersection_over_union(uint16(bbox),uint16(bboxT));
               if(iou > MaxIou)
                   MaxIou=iou;
                   indexMaxIou=indiciBBoxDataset(t);
               end
            end
            if(indexMaxIou > 0)%ci serve per distinguere i casi in cui venga detectato un oggetto non presente
                if(MaxIou > 0.5)&&(label == testTbl.Properties.VariableNames(indexMaxIou))%qui invece identifichiamo una detection non soddisfacente, ovvero o troppo spostata rispetto alla gtruth o con un'altra label
                    tp=tp+1;
                    predictedlabels=[predictedlabels label];
                    truelabels=[truelabels testTbl.Properties.VariableNames(indexMaxIou)];
                else
                    fp=fp+1;
                    predictedlabels=[predictedlabels label];
                    truelabels=[truelabels testTbl.Properties.VariableNames(indexMaxIou)];
                end
            else
                fp=fp+1;
            end
            
        end
         while(numLabelsPresent>numLabelsEffective)%calcoliamo i casi in cui è presente un oggetto ma non è stato riconosciuto
             fn=fn+1;
             numLabelsEffective=numLabelsEffective +1;
         end  
    end
    
end
A=[tp fp;fn -1];
[ap,recall,precision] = evaluateDetectionPrecision(results,testData);
%Fscore=2*((precision * recall)/(precision + recall));
M=confusionchart(truelabels,cellstr(predictedlabels),'RowSummary','row-normalized','ColumnSummary','column-normalized');
             
            
            
            
            
            
            