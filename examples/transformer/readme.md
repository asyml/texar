# for en-vi task

The task is IWSLT'15 English-Vietnamese dataset. For more information, please refer to https://nlp.stanford.edu/projects/nmt/

## obtain the dataset
```
mkdir data/en_vi
cp your train.en train.vi dev.en dev.vi test.en test.vi into this directory.
```
Feel free to try on different datasets as long as they are parallel text corpora and the file paths are set correctly.

## preprocessing the dataset and generate encoded vocabulary
```
bash preprocess_data.sh en vi
```

## training and evaluating the model
```
bash run_model.sh 1 train_and_evaluate en vi
bash run_model 1 test en vi
bash test_output.sh en vi
```
The `1` indicates one hparams set for en-vi task: `max_train_epoch=70 max_training_steps=125000 batch_size=2048 test_batch_size=64 beam_width=5 alpha=0.6 ...`. Read the `run_model.sh` for more details.


# we also give a sample script for wmt14 en-de task here.

## Obtain the dataset
```
#change the DOWNLOADED_DATA_DIR in the wmt14_en_de.sh to your own path.
bash wmt14_en_de.sh
```
You will obtain the dataset in the `./data/en_de/` directory

## preprocessing the dataset and generate encoded vocabulary
```
bash preprocess_data.sh en de
```
You will obtain the processed dataset in `./temp/data/run_en_de_wpm/data/` directory

## training the model

```
bash run_en_vi.sh 2 train_and_evaluate en de
```
Here `2` denotes one hparams set for wmt14 en-de task.

## test and evaluation
```
bash run_en_vi.sh 2 test en de
bash test_output.sh en de
```
