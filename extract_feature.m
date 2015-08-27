
%% Prepare environment
fprintf('------------------\nStage1: Preparing...\n------------------\n');
tic
if ~exist('prepared', 'var')
    clear mex;
    run('../../startup');
    gpu_id = 0;
    batch_per_gpu = 16;
    n_gpu = numel(gpu_id);
    batch_num = n_gpu*batch_per_gpu;
    DNN.caffe_mex('set_device_solver', gpu_id);
    DNN.caffe_mex('init_solver', 'get_fc7.prototxt', 'ZF5_iter110000', 'log\');
    root = 'D:\YuLiu\PersonIdentification\';
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
fprintf('------------------\nStage2: Extracting feature...\n------------------\n');
fprintf('For test set...\n');
num = numel(pdata.label_test);
feature.test = zeros(num,4096,'single');
max_iter = ceil(num / batch_num);
test_batch = cell(n_gpu, 1);
for i = 1 : max_iter
    batch_id = mod((i-1)*batch_num:i*batch_num-1, size(pdata.data_test, 4))+1;
    if mod(i*10/max_iter,max_iter)<1
        fprintf('%d/%d...\n',batch_id(1),num);
    end
    for i = 1:n_gpu
        gpu_batch_id = batch_id((i-1)*batch_per_gpu+1:i*batch_per_gpu);
        test_batch{i}{1} = single(pdata.data_test(:,:,:,gpu_batch_id)./255.0);
        test_batch{i}{1} = bsxfun(@minus,test_batch{i}{1},single(meanmat));
        test_batch{i}{2} = single(pdata.label_test(gpu_batch_id));
    end
    DNN.caffe_mex('test', test_batch);
    output = DNN.caffe_mex('get_response_solver', 'fc7');
    output = reshape(output{1},[4096,batch_num])';
    feature.test(batch_id,:) = output;
end

% DNN.caffe_mex('snapshot', 'model_final');
% save train_test_err.mat all_train_loss all_test_err -v7.3;





