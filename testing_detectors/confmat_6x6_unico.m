%script per la creazione di matrici 6x6
clear all
close all
%caricare la tabella dove sono presenti le immagini da testare. La tabella
%deve avere una prima colonna dove è presente il nome del file e le altre
%colonne coincidenti con le classi dell'oggetto yolo caricato.
load('beckmann_demoTbl');
load ('yoloBeckmann.mat');
testTbl=beckmann_demoTbl;
%si creano i datastore per effettuare il calcolo dei results.
imdsTest = imageDatastore(testTbl.imageFilename);
bldsTest = boxLabelDatastore(testTbl(:, 2:end));
testData = combine(imdsTest, bldsTest);
results = detect(yolov3Detector,testData,'MiniBatchSize',16);
%gli array predictedlabels e truelabels sono creati in quanto sono il
%parametro da passare alla funzione per ottenere la matrice di confusione.
fp=0; fn=0; tp=0; predictedlabels=[]; truelabels=[];

for i=1:height(results)
    %andiamo ad estrarre dalla riga i-esima tutte le label predette
    labels=results(i,'Labels');
    labels=labels{1,1};
    labels=labels{1,1};
    
    if(~isempty(labels))
        %se è stata predetta almeno una label, andiamo a prendere l'indice
        %di quella con score massimo
        [~,idx]=max(cell2mat(table2cell(results(i,'Scores'))));
        
        bboxes=results(i,'Boxes');
        bboxes=bboxes{1,1};
        bboxes=bboxes{1,1};
        %dopo aver messo in bboxes tutte le bbox predette, andiamo ad
        %estrarre la bbox e la label relativa allo score massimo
        %identificato in precedenza
        label=labels(idx,1);
        bbox=bboxes(idx,:);
        
        %partendo dalla colonna 2, andiamo a scorrere tutta la tabella dove
        %sono presenti le bbox manualmente apposte finchè non troviamo una
        %entrata non vuota. L'ipotesi di fondo è che esiste in ogni scena
        %una unica bbox da riconoscere e non ci sono scene senza alcuna
        %bbox.
        j=2;
        while(isempty(cell2mat(table2cell(testTbl(i,j)))))
            j=j+1;
        end
        %prendiamo la bbox che effettivamente ho messo manualmente, la gtruth
        bboxT=testTbl(i,j); 
        bboxT=cell2mat(table2cell(bboxT));
        if(~isempty(bboxT)) %condizione sempre verificata per le nostre ipotesi
            %calcolo della iou tra la bbox predetta e la gTruth
            iou=bb_intersection_over_union(uint16(bbox),uint16(bboxT));
        end
        
        if(iou<0.3)
            fp=fp+1; %o è errata localizzazione della bbox e label, oppure solo
                     % la localizzazione, è comunque un falso positivo.
        else
            %se la posizione è giusta, sicuramente questa predizione farà
            %parte della 6x6 finale e costruisco i due array,
            %predictedlabels e truelabels che servono per costruirla.
            predictedlabels=[predictedlabels label];
            truelabels=[truelabels testTbl.Properties.VariableNames(j)];
            
            if(strcmp(cellstr(label),testTbl.Properties.VariableNames(j)))
                tp=tp+1; %se la label è giusta, true positive.
            else
                fp=fp+1; % se la posizione è giusta ma label sbagliata è comunque 
                         % un false positive.
            end
        end
    else
        fn=fn+1;      %falso negativo, il modello non ha predetto alcuna bbox ma per
                      %le ipotesi fatte nella ground Truth c'è sempre
                      %una bbox.
    end
end
%creiamo le matrici 2x2 e 6x6.
A=[tp fp;fn -1];
M=confusionchart(truelabels,cellstr(predictedlabels),'RowSummary','row-normalized','ColumnSummary','column-normalized');