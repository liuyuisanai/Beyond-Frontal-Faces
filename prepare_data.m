function pdata = prepare_data(anno, para)
    if nargin < 2
        para.width = 2;
        para.height = 5;
    end
    if exist('cache/data.mat', 'file')
        load('cache/data.mat');
    else
        tridx   = anno.train_idx;
        teidx   = anno.test_idx;
        vaidx   = anno.val_idx;
        labels  = anno.identity_ids;
        hbbox   = anno.head_boxes;
        filename    = cellfun(@(x, y) [x '_' y '.jpg'], anno.photoset_ids, anno.photo_ids,'UniformOutput',false);
        num_train   = numel(tridx);
        num_test    = numel(teidx);
        num_val     = numel(vaidx);
        data_train  = zeros(224, 224, 3, num_train, 'uint8');
        data_test   = zeros(224, 224, 3, num_test, 'uint8');
        data_val    = zeros(224, 224, 3, num_val, 'uint8');
        pdata.label_train = labels(tridx);
        pdata.label_test  = labels(teidx);
        pdata.label_val   = labels(vaidx);
        fprintf('Preparing training set... ');
        parfor i = 1 : num_train
            idx = tridx(i);
            img = imread(fullfile(anno.dir_tr, filename{idx}));
            u = max(hbbox(idx, 2), 1);
            l = max(hbbox(idx, 1)-(para.width-1)/2*hbbox(idx, 3), 1);
            d = min(u+para.height*hbbox(idx, 4), size(img, 1));
            r = min(hbbox(idx, 1)+(para.width+1)/2*hbbox(idx, 3), size(img, 2));
            roi = imresize(img(u:d,l:r,:), [224, 224]);
            if size(roi, 3) < 3
                roi = repmat(roi, [1,1,3]);
            end
            data_train(:,:,:,i) = roi;
        end
        pdata.data_train = data_train;
        fprintf('done.\n');
        fprintf('Preparing testing set... ');
        parfor i = 1 : num_test
            idx = teidx(i);
            img = imread(fullfile(anno.dir_te, filename{idx}));
            u = max(hbbox(idx, 2), 1);
            l = max(hbbox(idx, 1)-(para.width-1)/2*hbbox(idx, 3), 1);
            d = min(u+para.height*hbbox(idx, 4), size(img, 1));
            r = min(hbbox(idx, 1)+(para.width+1)/2*hbbox(idx, 3), size(img, 2));
            roi = imresize(img(u:d,l:r,:), [224, 224]);
            if size(roi, 3) < 3
                roi = repmat(roi, [1,1,3]);
            end
            data_test(:,:,:,i) = roi;
        end
        pdata.data_test = data_test;
        fprintf('done.\n');
        fprintf('Preparing validation set... ');
        parfor i = 1 : num_val
            idx = vaidx(i);
            img = imread(fullfile(anno.dir_val, filename{idx}));
            u = max(hbbox(idx, 2), 1);
            l = max(hbbox(idx, 1)-(para.width-1)/2*hbbox(idx, 3), 1);
            d = min(u+para.height*hbbox(idx, 4), size(img, 1));
            r = min(hbbox(idx, 1)+(para.width+1)/2*hbbox(idx, 3), size(img, 2));
            roi = imresize(img(u:d,l:r,:), [224, 224]);
            if size(roi, 3) < 3
                roi = repmat(roi, [1,1,3]);
            end
            data_val(:,:,:,i) = roi;
        end
        pdata.data_val = data_val;
        fprintf('done.\n');
        fprintf('Saving to cache... ');
        save('cache/data.mat', 'pdata', '-v7.3');
        fprintf('done.\n');
    end
end

