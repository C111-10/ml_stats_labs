fis = mamfis("Name","creditResult");
fis = addInput(fis, [150 200], "Name", "score");
fis = addInput(fis, [0.1 1], "Name", "ratio");
fis = addInput(fis, [0 10], "Name", "credit");

fis = addMF(fis,"score","smf",[175 190],"Name","high");
fis = addMF(fis,"score","zmf",[155 175],"Name","low");

fis = addMF(fis,"ratio","zmf",[0.3 0.42],"Name","goodr");
fis = addMF(fis,"ratio","smf",[0.44 0.7],"Name","badr");

fis = addMF(fis,"credit","trapmf",[0 0 2 5],"Name","goodc");
fis = addMF(fis,"credit","trapmf",[5 8 10 10],"Name","badc");

fis = addOutput(fis,[0 10],"Name","decision");
fis = addMF(fis,"decision","trapmf",[5 8 10 10],"Name","Approve");
fis = addMF(fis,"decision","trapmf",[0 0 2 5],"Name","Reject");

rule1 = "score==high & ratio==goodr & credit==goodc => decision=Approve";
rule2 = "score==low & ratio==badr => decision=Reject";
rule3 = "score==low & credit==badc => decision=Reject";
ruleList = [rule1 rule2 rule3];

fis = addRule(fis, ruleList);

evalfis(fis,[190 0.39 1.5]);

sugenoFIS = convertToSugeno(fis);

subplot(2,2,1)
gensurf(fis)
title('Mamdani system (Output 1)')
subplot(2,2,2)
gensurf(sugenoFIS)
title('Sugeno system (Output 1)')