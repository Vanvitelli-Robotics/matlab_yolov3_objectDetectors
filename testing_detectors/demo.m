%va importato sia il detector desiderato che il file video
load('yoloDenkmit.mat');
obj = VideoReader('demoCompleta.mp4');
vid = read(obj);

 % read the total number of frames
frames = obj.NumFrames;
  
% file format of the frames to be saved in
 ST='.jpg';
% reading and writing the frames 
for x = 1 : 3 : frames
  %queste righe commentate consistono nell'andare ad inserire all'interno
  %dell'immagine tutte le bbox predette, senza filtrare e considerare solo
  %quella con score massimo.
%     % converting integer to string
%     Sx = num2str(x);
%   
%     % concatenating 2 strings
%     Strc = strcat(Sx, ST);
%     Vid = vid(:, :, :, x);
%     [bbox,score,label]=detect(yolov3Detector,Vid);
%     if(~isempty(label))
%         label=char(label);
%         score=string(score);
%         labels=strcat(label,'__:',score);
%         labels=cellstr(labels);
%         Vid=insertObjectAnnotation(Vid,'rectangle',bbox,labels);
%     end
%     %exporting the frames
%      imwrite(Vid, Strc);
%     
    
    % converting integer to string
    Sx = num2str(x);
  
    % concatenating 2 strings
    Strc = strcat(Sx, ST);
    Vid = vid(:, :, :, x);
    [bbox,score,label]=detect(yolov3Detector,Vid);
    if(~isempty(label))
        %label=char(label);
        [M,k]=max(score);
        label=label(k,1);
        label=char(label);
        score=string(M);
        labels=strcat(label,'__:',score);
        labels=cellstr(labels);
        bbox=bbox(k,:);
        Vid=insertObjectAnnotation(Vid,'rectangle',bbox,labels);
    end
    %exporting the frames
    imwrite(Vid, Strc);
    
end