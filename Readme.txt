To train and then then test the model, run the following command from the base directory:

python -m nmt.nmt --vocab_prefix=nmt/nmt_data/vocab  --train_prefix=nmt/nmt_data/train  --dev_prefix=nmt/nmt_data/dev  --test_prefix=nmt/nmt_data/test  --out_dir=nmt/nmt_model_2 --num_train_steps=12000  
--steps_per_stats=100 --num_layers=2  --num_units=128  --dropout=0.2 --metrics=bleu --attention_architecture=joint

To create dataset, use the file create_data.py in nmt/nmt_data directory which creates 2=the source and target vocabularies, train and test data files. Parameters in this file can be changed to create different data sets

Currently a partially trained model (3000 steps) on the synthetic data set using scaled_luong attention mechanism is stored in the nmt/nmt_model directory