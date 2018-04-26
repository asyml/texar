we give a sample for wmt14 en-de task here.

# Obtain the dataset
```
#change the DOWNLOADED_DATA_DIR in the wmt14_en_de.sh to your own path.
bash wmt14_en_de.sh
```
You will obtain the dataset in the `./data/en_de/` directory

# preprocessing the dataset and generate encoded vocabulary
```
bash preprocess_data.sh en de
```
You will obtain the processed dataset in `./temp/data/run_en_de_wpm/data/` directory

# training the model

```
bash run_en_vi.sh 200 train_and_evaluate en de
```
Here `200` denotes one hparams set for wmt14 en-de task.

# test and evaluation
```
bash run_en_vi.sh 200 test en de
bash test_output.sh en de
```
