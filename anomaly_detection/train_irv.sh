python runner.py --config-name train_dl model=cnn dataset=PoolData hyp.batch_size=32 hyp.num_epochs=50 dataset.num_features=36 dataset.preproc.part_trans_acc=True
python runner.py --config-name train_dl model=cnn_lw dataset=PoolData hyp.batch_size=32 hyp.num_epochs=50 dataset.num_features=36 dataset.preproc.part_trans_acc=True
python runner.py --config-name train_dl model=cnn_dn dataset=PoolData hyp.batch_size=32 hyp.num_epochs=50 dataset.num_features=36 dataset.preproc.part_trans_acc=True
python runner.py --config-name train_dl model=cnn_cw dataset=PoolData hyp.batch_size=32 hyp.num_epochs=50 dataset.num_features=36 dataset.preproc.part_trans_acc=True

python runner.py --config-name train_dl model=attention task=imputation dataset=PoolData hyp.batch_size=32 hyp.num_epochs=300 dataset.num_features=36 dataset.preproc.part_trans_acc=True
python runner.py --config-name train_dl model=attention task=classification dataset=PoolData hyp.batch_size=32 hyp.num_epochs=50 model.weights_fp=weights/PoolData/attention_model_weights_imputation.pth dataset.num_features=36 dataset.preproc.part_trans_acc=True