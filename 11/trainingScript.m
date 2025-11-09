% TRAINING & MODEL GENERATION SCRIPT - SCRIPT #1

% reading training data file
data=readtable('train.csv');
dataArr=table2array(data);
datain=dataArr(:,1:size(dataArr,2)-1);
dataout=dataArr(:,size(dataArr,2));

% running subtractive clustering and generate the fis model
fisOpt = genfisOptions("SubtractiveClustering",...
    "ClusterInfluenceRange",0.5);
fis = genfis(datain,dataout,fisOpt);

% print out the fis model
writeFIS(fis,'fisFile')

% evaluate the fis model
fuzout = evalfis(fis,datain);

% printing out the predictions
T = array2table(fuzout);
writetable(T, 'fuzOut.csv');

% calculate RMSE
trnRMSE = norm(fuzout-dataout)/sqrt(length(fuzout));

% print out the RMSE
fileID=fopen('trainRMSE-result.txt','wt');
fprintf(fileID, '%e\n', trnRMSE);
fclose(fileID);