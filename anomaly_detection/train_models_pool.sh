# Train the cnn model on all datasets
python runner.py --config-name train_dl model=cnn dataset=PoolData hyp.batch_size=8 hyp.num_epochs=20
python runner.py --config-name train_dl model=cnn dataset=PoolData hyp.batch_size=16 hyp.num_epochs=20
python runner.py --config-name train_dl model=cnn dataset=PoolData hyp.batch_size=32 hyp.num_epochs=20

# # Train the cnn_lw model on all datasets
# python runner.py --config-name train_dl model=cnn_lw dataset=PoolData hyp.batch_size=8 hyp.num_epochs=3
# python runner.py --config-name train_dl model=cnn_lw dataset=PoolData hyp.batch_size=16 hyp.num_epochs=3
# python runner.py --config-name train_dl model=cnn_lw dataset=PoolData hyp.batch_size=32 hyp.num_epochs=3

# # Train the cnn_dn model on all datasets
# python runner.py --config-name train_dl model=cnn_dn dataset=PoolData hyp.batch_size=8 hyp.num_epochs=3
# python runner.py --config-name train_dl model=cnn_dn dataset=PoolData hyp.batch_size=16 hyp.num_epochs=3
# python runner.py --config-name train_dl model=cnn_dn dataset=PoolData hyp.batch_size=32 hyp.num_epochs=3

# # Train the cnn_cw model on all datasets
# python runner.py --config-name train_dl model=cnn_cw dataset=PoolData hyp.batch_size=8 hyp.num_epochs=3
# python runner.py --config-name train_dl model=cnn_cw dataset=PoolData hyp.batch_size=16 hyp.num_epochs=3
# python runner.py --config-name train_dl model=cnn_cw dataset=PoolData hyp.batch_size=32 hyp.num_epochs=3

# # Train the time series forest model on all datasets
# python runner.py --config-name train_dl model=tsf dataset=PoolData

# # Train the attention model on all datasets (Pretrain first)
# python runner.py --config-name train_dl model=attention task=classification dataset=PoolData hyp.batch_size=8 hyp.num_epochs=50 hyp.learning_rate=1e-5
# python runner.py --config-name train_dl model=attention task=classification dataset=PoolData hyp.batch_size=16 hyp.num_epochs=50 hyp.learning_rate=1e-5
# python runner.py --config-name train_dl model=attention task=classification dataset=PoolData hyp.batch_size=32 hyp.num_epochs=50 hyp.learning_rate=1e-5

