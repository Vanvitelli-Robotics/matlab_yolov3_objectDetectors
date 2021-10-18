% import the video file
obj = VideoReader('demo per rete completa.mp4');
vid = read(obj);
load('yoloCompletoFixed.mat');
 % read the total number of frames
frames = obj.NumberOfFrames;
  
% file format of the frames to be saved in
 ST='.jpg';
% reading and writing the frames 
oggetti=["Beckmann";"Balea";"Denkmit";"Finish"];%mi serve per controllare che solo 1 bbox per ogi pggetto venga generata
for y = 1 : 3 : frames
    
    % converting integer to string
    Sx = num2str(y);
  
    % concatenating 2 strings
    Strc = strcat(Sx, ST);
    Vid = vid(:, :, :, y);
    [bboxes,scores,labels]=detect(yolov3Detector,Vid);
    if(~isempty(labels))
        
        %label=char(label);
        %[M,k]=max(score);
        
      for n=1:4 %scorro tutti gli oggetti ed effettuo il cotrollo
        
        dim=size(labels);
        numLabelsPredicted=dim(1);%ricavo il nuemro di label generate 
           
        indiciLabel=[];
        indexmax=0; 
        numRipetizioni=0;%indica quante volte viene prodotta la label di uno stesso oggetto
        
        for j=1:numLabelsPredicted
              y=strfind(string(labels(j,1)),oggetti(n,1));%ad ogni iterazone del ciclo n, verifico in quali label è presente il nome dell oggetto n-esimo
              if(size(y)>0)%vado qui solo le la label contiene quel nome
                  indiciLabel=[indiciLabel j];%salvo gli indici delle label prese
                  y=[];
              end
        end
        
        numRipetizioni=size(indiciLabel);
        numRipetizioni=numRipetizioni(2);%ricavo il nuemro di indici e quindi di ripetizioni
        
        if(numRipetizioni>1)%ovvero se c'è più di una label riferita allo stesso oggetto
            
            %score=cell2mat(results.Scores(i));%salvo gli score delle bbox 
           
            indexmax=0;
            maxscore=0;
            
            for j=1:numRipetizioni
                if(maxscore < scores(indiciLabel(1,j),1))
                   indexmax=indiciLabel(1,j);%salvo l'indice dello score massimo
                   maxscore=scores(indiciLabel(1,j),1);%agiorno lo score massimo
                end
            end
            
            for j=1:numRipetizioni
                if(~(indiciLabel(1,j)==indexmax))
                    labels(indiciLabel(1,j),:)=[];%elimino le label che non corrispondono allo score massimo
                    bboxes(indiciLabel(1,j),:)=[];%elimino le bbox che non corrispondono allo score massimo
                    scores(indiciLabel(1,j))=[];%elimino lo score
                end
             end
                
            end
        
      end
      for k=1:numLabelsPredicted 
        etichetta=labels(k,1);
        etichetta=char(etichetta);
        score=string(scores(k));
        label=strcat(etichetta,'__:',score);
        label=cellstr(label);
        bbox=bboxes(k,:);
        Vid=insertObjectAnnotation(Vid,'rectangle',bbox,label);
      end
    end
    %exporting the frames
    imwrite(Vid, Strc);
    
end

