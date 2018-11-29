This gives the explanation on data preparation.

When you run `data/download_glue_data.py` in the parent directory, by default, all datasets in GLEU will be stored here. For more information on GLUE, you can refer to 
[gluebenchmark](https://gluebenchmark.com/tasks)

Here we show the data format of the SSN-2 example.

```
# train sample in SST-2/train.tsv
sentence	label
hide new secretions from the parental units 	0
contains no wit , only labored gags 	0

# evaluate sample in SST-2/dev.tsv
sentence	label
it 's a charming and often affecting journey . 	1
unflinchingly bleak and desperate 	0

# test sample in SST-2/test.tsv
index	sentence
0	uneasy mishmash of styles and genres .
1	this film 's relationship to actual tension is the same as what christmas-tree flocking in a spray can is to actual snow : a poor -- if durable -- imitation .
```

In SST dataset, The train data and evaluation data are in the same schema. The first line gives the header information, sentence and label. In the following lines, the sentence is a space-seperated string, and the label is 0 or 1. While the test data has a different schema, where the first column is a unique index for each test example, the second column is the space-seperated string.

In the `bert/utils/data_utils`, there are five types of Data Processor Implemented. You can run `python bert_classifier_main.py` and specify the `--task` to run on different datasets.
