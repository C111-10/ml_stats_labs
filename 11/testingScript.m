% TESTING SCRIPT - SCRIPT #2

% second script for the test
fis = readfis('fisFile.fis');

% reading test data file
data=readtable('test.csv');
dataArr=table2array(data);
valdatain=dataArr(:,1:size(dataArr,2)-1);
valdataout=dataArr(:,size(dataArr,2));

% evaluate our fis model with test data
valfuzout = evalfis(fis,valdatain);
valRMSE = norm(valfuzout-valdataout)/sqrt(length(valfuzout));

% printing out the predictions
T = array2table(valfuzout);
writetable(T, 'testfuzOut.csv');

% print out the RMSE
fileID=fopen('testRMSE-result.txt','wt');
fprintf(fileID, '%e\n', valRMSE);
fclose(fileID);
