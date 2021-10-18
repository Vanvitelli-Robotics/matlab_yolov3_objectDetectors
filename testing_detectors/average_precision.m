%script per il calcolo delle average precision, recall, precision
load('tabelle_singole_demoCompleta.mat');
testTbl=baleaTbl;
imdsTest = imageDatastore(testTbl.imageFilename);
bldsTest = boxLabelDatastore(testTbl(:, 2:end));
testData = combine(imdsTest, bldsTest);
results = detect(yolov3Detector,testData,'MiniBatchSize',16);
[ap,recall,precision] = evaluateDetectionPrecision(results,testData);
