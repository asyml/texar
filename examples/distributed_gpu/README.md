# Distributed Version of Language Model on PTB

This is the distributed version of language model on PTB with Horovod framework.While the training phrase is distributed over multiple GPUs, the evaluation and testing phrase are done only on single GPU.

## Prerequisites

To install horovod, you need to install OpenMPI first. It's recommended that the version of openMPI is larger than `3.0.0`, you can run `mpirun --version` to check the version. If it's elder than `3.0.0`, you can run the following command to install the OpenMPI 4.0.0.

```
# download and install OpenMPI
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.0.tar.gz
tar xvf openmpi-4.0.0.tar.gz
cd openmpi-4.0.0/
./configure --prefix=/usr/local/openmpi
sudo make all install

# Add the path of installed openMPI to your system path
export PATH=/usr/local/openmpi/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/openmpi/lib:$LD_LIBRARY_PATH
```

Then you can install horovod with

```
pip install horovod
```

## Highlights

Compared to the single-machine version, there are a few things different:

- Basic setting of Horovod in your codes. Generally, you will need in insert the horovod wrappers in appropriate portion of your codes.
    1. [`hvd.init()`](https://github.com/asyml/texar/blob/master/examples/distributed_gpu/lm_ptb_distributed.py#L76): initialize the horovod
    2. [`hvd.DistributedOptimizer`](https://github.com/asyml/texar/blob/master/examples/distributed_gpu/lm_ptb_distributed.py#L131): wrap your optimizer.
    3. [`hvd.broadcast_global_variables(0)`](https://github.com/asyml/texar/blob/master/examples/distributed_gpu/lm_ptb_distributed.py#L194): set the operator to broadcast your global variables to different processes from rank-0 process.
    4. [set visible GPU list](https://github.com/asyml/texar/blob/master/examples/distributed_gpu/lm_ptb_distributed.py#L194) by `config.gpu_options.visible_device_list = str(hvd.local_rank())`, to make each process see the attached single GPU.
    5. [run the broadcast node](https://github.com/asyml/texar/blob/master/examples/distributed_gpu/lm_ptb_distributed.py#L194): run the broadcast operator before training
- Data feeding:
    - You should [split your dataset into shards](https://github.com/asyml/texar/blob/master/examples/distributed_gpu/lm_ptb_distributed.py#L194) before sending them to different processes, to make sure different GPUs are fed different mini-batch in each iteration.
    - Because we update the global variables based on the mini-batches in different processes, we may need to adjust the `learning rate`, `batch_size` to fit the distributed settings. In this example, we [scale down the specified `batch_size` with the number of processes](https://github.com/asyml/texar/blob/master/examples/distributed_gpu/ptb_reader.py#L45) before feeding the mini-batch into the graph, to replicate the gradient computation in single-gpu setting.

## Usage ##

Run the following command to run on different GPUs where each GPU is used by one process.
```
mpirun -np 2 \
    -H  server1:1,server2:1\
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl tcp,self \
    -mca btl_tcp_if_include ens3 \
    python lm_ptb_distributed.py [--config config_small] [--data_path ./]
```

Here:
  * `-np`: number of processes
  * `-H`: specifies the address of different servers and the number of processes used in each server.
  * `--bind-to none`: specifies Open MPI to not bind a training process to a single CPU core (which would hurt performance).
  * `-map-by slot`: allows you to have a mixture of different NUMA configurations because the default behavior is to bind to the socket.
  * `-mca`: set the MPI communication interface. Use the setting specified above to avoid many multiprocessing and network communication issues.
  * `-x`: to specify (-x NCCL_DEBUG=INFO) or copy (-x LD_LIBRARY_PATH) an environment variable to all the workers.
  * `--config` specifies the config file to use. E.g., the above use the configuration defined in config_small.py
  * `--data_path` specifies the directory containing PTB raw data (e.g., ptb.train.txt). If the data files do not exist, the program will automatically download, extract, and pre-process the data.

The model will begin training, and will evaluate on the validation data periodically, and evaluate on the test data after the training is done. 

## Results ##

We test the performance on two AWS p2.xlarge instances. 
Since the language model is just a small example and the communication cost is considerable, it doesn't scale well in 2-GPU 2-machine in terms of speedup rate. But the performance in multi-gpu is the same as that of the single-gpu environment, which is shown as follows:

| config | epochs | train | valid  | test  | time/epoch (2-gpu) | time/epoch(single-gpu) |
| -------| -------| ------| -------| ------| -----| -----|
| small  | 13     | 40.81 | 118.99 | 114.72| 207s | 137s |
| medium | 39     | 44.18 |  87.63 |  84.42| 461s | 311s |
| large  | 55     | 36.54 |  82.55 |  78.72| 1765s | 931s |

You can refer to the corresponding single-gpu version of this example [here](https://github.com/asyml/texar/tree/master/examples/language_model_ptb).
