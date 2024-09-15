# Train the cnn model on all datasets
python runner.py --config-name train_dl model=cnn dataset=BasicMotions hyp.batch_size=8 hyp.num_epochs=3
python runner.py --config-name train_dl model=cnn dataset=ChestMntdAcl hyp.batch_size=32 hyp.num_epochs=3
python runner.py --config-name train_dl model=cnn dataset=Epilepsy hyp.batch_size=16 hyp.num_epochs=3
python runner.py --config-name train_dl model=cnn dataset=WalkingSittingStanding hyp.batch_size=32 hyp.num_epochs=3

# Train the cnn_lw model on all datasets
python runner.py --config-name train_dl model=cnn_lw dataset=BasicMotions hyp.batch_size=8 hyp.num_epochs=3
python runner.py --config-name train_dl model=cnn_lw dataset=ChestMntdAcl hyp.batch_size=32 hyp.num_epochs=3
python runner.py --config-name train_dl model=cnn_lw dataset=Epilepsy hyp.batch_size=16 hyp.num_epochs=3
python runner.py --config-name train_dl model=cnn_lw dataset=WalkingSittingStanding hyp.batch_size=32 hyp.num_epochs=3

# Train the cnn_dn model on all datasets
python runner.py --config-name train_dl model=cnn_dn dataset=BasicMotions hyp.batch_size=8 hyp.num_epochs=3
python runner.py --config-name train_dl model=cnn_dn dataset=ChestMntdAcl hyp.batch_size=32 hyp.num_epochs=3
python runner.py --config-name train_dl model=cnn_dn dataset=Epilepsy hyp.batch_size=16 hyp.num_epochs=3
python runner.py --config-name train_dl model=cnn_dn dataset=WalkingSittingStanding hyp.batch_size=32 hyp.num_epochs=3

# Train the cnn_cw model on all datasets
python runner.py --config-name train_dl model=cnn_cw dataset=BasicMotions hyp.batch_size=8 hyp.num_epochs=3
python runner.py --config-name train_dl model=cnn_cw dataset=ChestMntdAcl hyp.batch_size=32 hyp.num_epochs=3
python runner.py --config-name train_dl model=cnn_cw dataset=Epilepsy hyp.batch_size=16 hyp.num_epochs=3
python runner.py --config-name train_dl model=cnn_cw dataset=WalkingSittingStanding hyp.batch_size=32 hyp.num_epochs=3

# Train the time series forest model on all datasets
python runner.py --config-name train_dl model=tsf dataset=BasicMotions
python runner.py --config-name train_dl model=tsf dataset=ChestMntdAcl
python runner.py --config-name train_dl model=tsf dataset=Epilepsy
python runner.py --config-name train_dl model=tsf dataset=WalkingSittingStanding

# Train the attention model on all datasets (Pretrain first)
python runner.py --config-name train_dl model=attention task=imputation dataset=BasicMotions hyp.batch_size=8 hyp.num_epochs=50
python runner.py --config-name train_dl model=attention task=classification dataset=BasicMotions hyp.batch_size=8 hyp.num_epochs=50 model.weights_fp=weights/BasicMotions/attention_model_weights_imputation.pth

python runner.py --config-name train_dl model=attention task=imputation dataset=ChestMntdAcl hyp.batch_size=32 hyp.num_epochs=50
python runner.py --config-name train_dl model=attention task=classification dataset=ChestMntdAcl hyp.batch_size=32 hyp.num_epochs=50 model.weights_fp=weights/ChestMntdAcl/attention_model_weights_imputation.pth

python runner.py --config-name train_dl model=attention task=imputation dataset=Epilepsy hyp.batch_size=16 hyp.num_epochs=50
python runner.py --config-name train_dl model=attention task=classification dataset=Epilepsy hyp.batch_size=16 hyp.num_epochs=50 model.weights_fp=weights/Epilepsy/attention_model_weights_imputation.pth

python runner.py --config-name train_dl model=attention task=imputation dataset=WalkingSittingStanding hyp.batch_size=32 hyp.num_epochs=50
python runner.py --config-name train_dl model=attention task=classification dataset=WalkingSittingStanding hyp.batch_size=32 hyp.num_epochs=50 model.weights_fp=weights/WalkingSittingStanding/attention_model_weights_imputation.pth