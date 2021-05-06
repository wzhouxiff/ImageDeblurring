model_path='./checkpoints'
model_name='DeblurOnly'

python -u test.py  \
--name test \
--dataset_mode GoproHDF5SingleTest2Bin --event_name  EventBin3 \
--sequence_num  1  \
--eventbins_between_frames 2 \
--test_batch_size 1  --n_threads 0  \
--model  $model_name \
--Gopro \
--load_G  $model_path'/weight.pth' \
--data_dir  testdata/ \
--output_dir $model_path'/results' \
--VerifyL2
