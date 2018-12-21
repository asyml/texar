# Distributed Version of Language Model on PTB

This is the distributed version of language model on PTB with Horovod framework.

Compared to the single-machine version, there are a few things different:

- Basic setting of Horovod in your codes. Generally, you will need in insert the horovod wrappers in appropriate portion of your codes
    - `hvd.init()`: initialize the horovod
    - set visible GPU list by `config.gpu_options.visible_device_list = str(hvd.local_rank())`, to make each process see the attached single GPU.
    - `hvd.DistributedOptimizer`: wrap your optimizer.
    - `hvd.broadcast_global_variables(0).run()` broadcast your global variables to different processes from rank-0 process.
- Data feeding:
    - As introduced in the [official Horovod Repository](https://github.com/uber/horovod/issues/223), you should split your dataset into shards before sending them to different processes, to make sure different GPUs are fed different mini-batch in each iteration.
    - Because we update the global variables based on the mini-batches in different processes, we may need to adjust the `learning rate`, `batch_size` to fit the distributed settings. In this example, we scale down the specified `batch_size` with the number of processes before feeding the mini-batch into the graph, to replicate the gradient computation in single-gpu setting.

# Language Model on PTB #

This example builds an LSTM language model, and trains on PTB data. Model and training are described in   
[(Zaremba, et. al.) Recurrent Neural Network Regularization](https://arxiv.org/pdf/1409.2329.pdf). This is a reimpmentation of the TensorFlow official PTB example in [tensorflow/models/rnn/ptb](https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb).

The example shows:
  * Contruction of simple model, involving the `Embedder` and `RNN Decoder`.
  * Use of Texar with external Python data pipeline ([ptb_reader.py](./ptb_reader.py)).
  * Specification of various features of train op, like *gradient clipping* and *lr decay*.

## Usage ##

By default, the following cmd trains a single-gpu model with Horovod:

```
python lm_ptb.py [--config config_small] [--data_path ./]
```

Here:
  * `--config` specifies the config file to use. E.g., the above use the configuration defined in [config_small.py](./config_small.py)
  * `--data_path` specifies the directory containing PTB raw data (e.g., `ptb.train.txt`). If the data files do not exist, the program will automatically download, extract, and pre-process the data.

The model will begin training, and will evaluate on the validation data periodically, and evaluate on the test data after the training is done. 

Run the following command to run on different GPUs where each GPU is used by one process.
```
mpirun -np 2 \
    -H  server1:1,server2:1\
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl tcp,self \
    -mca btl_tcp_if_include ens3 \
    python lm_ptb_distributed.py
```

## Results ##

As per the TensorFlow official PTB example, the perplexity of different configs is:

| config | epochs | train | valid  | test  |
| -------| -------| ------| -------| ------|
| small  | 13     | 37.99 | 121.39 | 115.91|
| medium | 39     | 48.45 |  86.16 |  82.07|
| large  | 55     | 37.87 |  82.62 |  78.29|
