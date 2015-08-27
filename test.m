label = pdata.label_test;
[~,mapid,ulabel] = unique(label);
id_tr = [];
id_te = [];
for i = 1 : numel(mapid)
    bottom = mapid(i);
    if i == numel(mapid)
        top = numel(ulabel);
    else
        top = mapid(i+1)-1;
    end
%     mid = floor((bottom+top)/2);
    mid = bottom+8;
    id_tr = [id_tr,bottom:mid];
    id_te = [id_te,mid+1:top];
end
tic
model = train(double(ulabel(id_tr)),sparse(double(feature.test(id_tr,:))),'-c 10 -e 0.1 -v 5 -q heart_scale');
toc
tic
[predict_label, acc, prob] = predict(double(ulabel(id_te)),sparse(double(feature.test(id_te,:))),model);
toc