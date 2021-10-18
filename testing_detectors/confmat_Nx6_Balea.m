%script per la creazione della matrice Nx6 per l'oggetto Balea
clear all
close all
%si va a caricare la tabella dove sono presenti le immagini da testare. La 
%tabella deve avere una prima colonna dove è presente il nome del file. Il 
%numero totale di colonne per la nostra applicazione sarà 25 e la tabella 
%caricata è la stessa nei 4 script per la creazione delle matrici Nx6.
load('DemoCompletetbl');
load ('yoloBalea.mat');
testTbl=DemoCompletetbl;
%si creano i datastore per effettuare il calcolo dei results.
imdsTest = imageDatastore(testTbl.imageFilename);
bldsTest = boxLabelDatastore(testTbl(:, 2:end));
testData = combine(imdsTest, bldsTest);
results = detect(yolov3Detector,testData,'MiniBatchSize',16);
%gli array predictedlabels e truelabels sono creati in quanto sono il
%parametro da passare alla funzione per ottenere la matrice di confusione.
fp=0; fn=0; tp=0; ct=0; predictedlabels=[]; truelabels=[];

for i=1:height(results)
    iouarray=[];
    indexarray=[];
    %si estraggono dalla riga i-esima dei results tutte le labels predette
    labels=results(i,'Labels');
    labels=labels{1,1};
    labels=labels{1,1};
    
    if(~isempty(labels))
        %se è stata predetta almeno una label, si va a prendere l'indice
        %di quella con score massimo
        
        [~,idx]=max(cell2mat(table2cell(results(i,'Scores'))));
        bboxes=results(i,'Boxes');
        bboxes=bboxes{1,1};
        bboxes=bboxes{1,1};
        %dopo aver messo nella variabile bboxes tutte le bbox predette, 
        %si estrae la bbox e la label relativa allo score massimo
        %tramite l'indice ricavato in precedenza 
        label=cellstr(labels(idx,1));
        bbox=bboxes(idx,:);
        
        %scorrendo tutte le colonne della i-esima riga della tabella dove
        %sono presenti le bbox apposte manualmente (testTbl), si va a
        %creare un array contenente tutte le iou tra la bbox predetta e 
        %quelle presenti, e un array con gli indici che determinano il
        %numero della colonna dove sono presenti tali bbox. L'ipotesi alla 
        %base è che ci sia una bbox per ogni oggetto in ogni immagine.
        for j=2:width(testTbl)
            if(~isempty(cell2mat(table2cell(testTbl(i,j)))))
                iouarray=[iouarray bb_intersection_over_union(uint16(bbox),uint16(cell2mat(table2cell(testTbl(i,j)))))];
                indexarray=[indexarray j];
            end
        end
        %successivamente si estrae la iou massima tra quelle calcolate in
        %precedenza e il numero di colonna - iouindex - relativo alla bbox 
        %con cui ciò accade.
        [iou, iouindex]=max(iouarray);
        index=indexarray(iouindex); %index ora è il numero di colonna in 
                                    %testTbl dove trovare la bbox con cui 
                                    %si ha iou massima
        %se la iou massima è pari a 0 vuol dire che è stato rilevato un
        %oggetto diverso dai 4 in esame.
        if(iou==0)
            predictedlabels=[predictedlabels label];
            truelabels=[truelabels "altro"];
            fp=fp+1;
            continue;
        end
        
        %se la iou>0 si può costruire immediatamente l'array delle predicted
        %labels. Successivamente si verifica se si tratta di un true
        %positive, e in caso non sia cosi si va a identificare con quale
        %oggetto sia avvenuta la confusione: ciò lo si fa andando a
        %considerare il nome della colonna identificata dalla variabile
        %index trovata in precedenza.
         predictedlabels=[predictedlabels label];
         
        if(iou>0.3 && strcmp(label,testTbl.Properties.VariableNames(index)))
            tp=tp+1; %true positive
            truelabels=[truelabels testTbl.Properties.VariableNames(index)];
        else
            if(contains(testTbl.Properties.VariableNames(index),"Beckmann"))
                truelabels=[truelabels "Beckmann"];
            elseif(contains(testTbl.Properties.VariableNames(index),"Denkmit"))
                truelabels=[truelabels "Denkmit"];
            elseif(contains(testTbl.Properties.VariableNames(index),"finish"))
                truelabels=[truelabels "Finish"];
            else %se nessuna delle precedenti è verificata, vuol dire che la
                 %iou massima (e >0) si è avuta con l'oggetto giusto ma label,
                 %e quindi orientamento, sbagliata.
                truelabels=[truelabels testTbl.Properties.VariableNames(index)];
            end
            fp=fp+1; %in ogni caso si tratta di un false positive
        end
    else
        fn=fn+1;  %falso negativo, il modello non ha predetto alcuna bbox 
                  %ma per le ipotesi fatte nella ground Truth c'è sempre
                  %una bbox
    end
end
%si crea infine la matrice Nx6 ordinando le classi in modo coerente con le 
%altre Nx6 cosi da avere un confronto visivo diretto. Gli elementi 
%appartenenti all'array order devono essere una permutazione dei nomi delle 
%classi inserite per la costruzione della confusion matrix, per cui è 
%necessario sapere quali classi compaiono tra le 4 aggiuntive (gli altri 3
%oggetti + "altro"). Ciò può essere ottenuto commentando la riga relativa
%all'ordinamento delle classi e successivamente creare l'array order di
%conseguenza.
A=[tp fp;fn 0];
order=["Balea_fronte","Balea_fronte_steso","Balea_fronte_steso_capovolto","Balea_retro","Balea_retro_steso","Balea_retro_steso_capovolto","Beckmann","Denkmit","Finish","altro"];
figure
M=confusionchart(cellstr(truelabels),predictedlabels,'ColumnSummary','column-normalized');
sortClasses(M,order);