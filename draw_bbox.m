function draw_bbox(anno, maxiter, para)
        if nargin < 3
            para.width = 2;
            para.height = 5;
        end
        hbbox   = anno.head_boxes;
        teidx   = anno.test_idx;
        filename    = cellfun(@(x, y) [x '_' y '.jpg'], anno.photoset_ids, anno.photo_ids,'UniformOutput',false);
        num_test    = numel(teidx);
        for i = 1 : max(num_test,maxiter)
            idx = teidx(i);
            img = imread(fullfile(anno.dir_te, filename{idx}));
            u = max(hbbox(idx, 2), 1);
            l = max(hbbox(idx, 1)-(para.width-1)/2*hbbox(idx, 3), 1);
            d = min(u+para.height*hbbox(idx, 4), size(img, 1));
            r = min(hbbox(idx, 1)+(para.width+1)/2*hbbox(idx, 3), size(img, 2));
            imshow(img);
            rectangle('Position', hbbox(idx, :));
            rectangle('Position', [l, u, r-l, d-u]);
            drawnow;
            saveas(gcf,['draw\' num2str(i) '.jpg']);
        end
        
end

