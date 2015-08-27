%% Prepare environment
fprintf('------------------\nStage1: Preparing...\n------------------\n');
tic
if ~exist('prepared', 'var')
    clear mex;
    clear iter te_acc te_loss tr_acc tr_loss;
    run('../../startup');
    gpu_id = 0:7;
    batch_per_gpu = 16;
    n_gpu = numel(gpu_id);
    DNN.caffe_mex('set_device_solver', gpu_id);
    DNN.caffe_mex('init_solver', 'solver_100k200k.prototxt', 'pretrained_model', 'log\');
    root = 'D:\dataset\PersonIdentification\';
    train_dir = fullfile(root, 'train');
    annotation_dir = fullfile(root, 'annotations');
    test_dir = fullfile(root, 'test');
    validate_dir = fullfile(root, 'val');
    load(fullfile(annotation_dir, 'data.mat'));
    data.dir_tr = train_dir;
    data.dir_te = test_dir;
    data.dir_val= validate_dir; 
    if ~exist('pdata', 'var')
        pdata = prepare_data(data);
    end
    prepared = 1; 
    iter_ = 1;
end
toc
fprintf('Prepare done.\n');

%% Start train
fprintf('------------------\nStage2: Training...\n------------------\n');
if exist('iter','var')
    fprintf('Exist iter, continue training at iter = %d...\n', iter);
    iter_ = iter;
else
    iter_ = 1;
    test_iter = 1;
    max_iter = 1000000;
    train_batch = cell(n_gpu, 1);
end

tic
for iter = iter_:max_iter
    drawnow;
    batch_num = n_gpu*batch_per_gpu;
    batch_id = randperm(numel(pdata.label_train), batch_num);
    for i = 1:n_gpu
        gpu_batch_id = batch_id((i-1)*batch_per_gpu+1:i*batch_per_gpu);
        train_batch{i}{1} = single(pdata.data_train(:,:,:,gpu_batch_id)./255.0);
        train_batch{i}{1} = bsxfun(@minus,train_batch{i}{1},single(meanmat));
        train_batch{i}{2} = single(pdata.label_train(gpu_batch_id));
    end
    ret = DNN.caffe_mex('train', train_batch);
    acc = ret(1).results;
    loss=ret(2).results;
    tr_acc(iter) = acc;
    tr_loss(iter) = loss;
    if mod(iter,10)<1
        fprintf('.')
    end
    %show train loss
    if mod(iter,100)<1 
        fprintf('\n');
        toc
        fprintf('iter %d: loss=%f acc=%f\n', iter, mean(tr_loss(iter-100+1:iter)), mean(tr_acc(iter-100+1:iter)));
        tic
    end
    %save result
    if mod(iter,5000)<1
        DNN.caffe_mex('snapshot', ['ZF5_iter' num2str(iter)]);
    end
    %test
%     if mod(iter,1000)<1
%         tic
%         test_iter_max = ceil(size(pdata.data_test, 4)/batch_num);
%         test_batch = cell(n_gpu, 1);
%         acc = 0;
%         loss = 0;
%         for i = 1 : test_iter_max
%             batch_id = mod((i-1)*batch_num:i*batch_num-1, size(pdata.data_test, 4))+1;
%             for i = 1:n_gpu
%                 gpu_batch_id = batch_id((i-1)*batch_per_gpu+1:i*batch_per_gpu);
%                 test_batch{i}{1} = single(pdata.data_test(:,:,:,gpu_batch_id)./255.0);
%                 test_batch{i}{1} = bsxfun(@minus,test_batch{i}{1},single(meanmat));
%                 test_batch{i}{2} = single(pdata.label_test(gpu_batch_id));
%             end
%             ret = DNN.caffe_mex('test', test_batch);
%             acc = acc+ret(1).results;
%             loss= loss+ret(2).results;
%         end
%         acc = acc / test_iter_max;
%         loss = loss / test_iter_max;
%         te_acc(test_iter) = acc;
%         te_loss(test_iter) = loss;
%         toc
%         fprintf('Test: loss:%f, acc:%f\n', loss, acc);
%         plot(te_acc,'g');
%         hold on;
%         plot(te_loss,'r');
%         hold off;
%         test_iter = test_iter + 1;
%     end
    plot(tr_loss,'r');
    hold on;
    plot(tr_acc,'g');
    hold off;
end

% DNN.caffe_mex('snapshot', 'model_final');
% save train_test_err.mat all_train_loss all_test_err -v7.3;





