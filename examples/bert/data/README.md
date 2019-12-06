This gives the explanation on data preparation.

When you run `data/download_glue_data.py` in the parent directory, by default, all datasets in the General Language Understanding Evaluation (GLUE) will be stored here. For more information on GLUE, please refer to 
[gluebenchmark](https://gluebenchmark.com/tasks)

Here we show the data format of the SSN-2 dataset.

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

* As above, in SST, the train/eval data are in the following format: the first line gives the header information, including `sentence` and `label`. In each of the following lines, the sentence is a space-seperated string, and the label is `0` or `1`. 
* The test data is in a different format: the first column is a unique index for each test example, the second column is the space-seperated string.


In [`bert/utils/data_utils.py`](https://github.com/asyml/texar/blob/master/examples/bert/utils/data_utils.py), there are 5 types of `Data Processor` implemented. You can run `python bert_classifier_main.py` and specify `--task` to run on different datasets.
