# Model Training with Multi/Distributed GPUs

This example shows how models built with Texar can be trained with multiple GPUs on single or multiple machines. Multi/Distributed-GPU training is based on the third-party library [Horovod](https://github.com/uber/horovod).

Here we take language model for example, adapting the [single-GPU language model example](https://github.com/asyml/texar/tree/master/examples/language_model_ptb) by adding a few lines of Horovod-related code to enable distributed training (more details below).

## Prerequisites

Two third-party packages are required:

* `openmpi >= 3.0.0`
* `horovod`

The following commands install [OpenMPI](https://www.open-mpi.org) 4.0.0 to the path `/usr/local/openmpi`. Run `mpirun --version` to check the version of installed OpenNMT.
```
# Download and install OpenMPI
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.0.tar.gz
tar xvf openmpi-4.0.0.tar.gz
cd openmpi-4.0.0/
./configure --prefix=/usr/local/openmpi
sudo make all install

# Add path of the installed OpenMPI to your system path
export PATH=/usr/local/openmpi/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/openmpi/lib:$LD_LIBRARY_PATH
```

Then install Horovod with the cmd:
```
pip install horovod
```

## Adapting Single-GPU Code for distributed Training

Based on the [single-GPU code](https://github.com/asyml/texar/tree/master/examples/language_model_ptb), we made the following adaptions. Note that one processor is created for each GPU.

- Setting up Horovod in the code (click the links below to see the corresponding actual code in `lm_ptb_distributed.py`):
    1. [`hvd.init()`](https://github.com/asyml/texar/blob/master/examples/distributed_gpu/lm_ptb_distributed.py#L76): initialize Horovod
    2. [`hvd.DistributedOptimizer`](https://github.com/asyml/texar/blob/master/examples/distributed_gpu/lm_ptb_distributed.py#L131): wrap your optimizer.
    3. [`hvd.broadcast_global_variables(0)`](https://github.com/asyml/texar/blob/master/examples/distributed_gpu/lm_ptb_distributed.py#L191): set the operator to broadcast your global variables to different processes from rank-0 process.
    4. [set visible GPU list](https://github.com/asyml/texar/blob/master/examples/distributed_gpu/lm_ptb_distributed.py#L194) by `config.gpu_options.visible_device_list = str(hvd.local_rank())`, to make each process see the attached single GPU.
    5. [run the broadcast node](https://github.com/asyml/texar/blob/master/examples/distributed_gpu/lm_ptb_distributed.py#L203): run the broadcast operator before training
- Data sharding:
    1. To make sure different GPUs (processors) receive different data batches in each iteration, we [shard the training data](https://github.com/asyml/texar/blob/master/examples/distributed_gpu/ptb_reader.py#L52) into `N` parts, where `N` is the number of GPUs (processors).
    2. In this example, `batch_size` in the config files denotes the total batch size in each iteration of all processors. That is, in each iteration, each processor receives `batch_size`/`N` data instances. This replicates the gradients in the single-GPU setting, and we use the same `learning_rate` as in single-GPU.

## Usage ##

Run the following command to train the model with multiple GPUs on multiple machines:
```
mpirun -np 2 \
    -H  [IP-adress-of-server1]:1,[IP-address-of-server2]:1\
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl tcp,self \
    -mca btl_tcp_if_include ens3 \
    python lm_ptb_distributed.py --config config_small --data_path ./
```

Here:
  * The key configurations for ordinary users:
  
      - `-np`: total number of processes
      - `-H`: IP addresses of different servers and the number of processes used in each server. For example, `-H 192.168.11.22:1,192.168.33.44:1`. To run on local machines, set, e.g., `-H localhost:2`.
      
  * Other advanced configurations:
  
      - `--bind-to none`: specifies OpenMPI to not bind a training process to a single CPU core (which would hurt performance).
      - `-map-by slot`: allows you to have a mixture of different NUMA configurations because the default behavior is to bind to the socket.
      - `-x`: specifies (`-x NCCL_DEBUG=INFO`) or copies (`-x LD_LIBRARY_PATH`) an environment variable to all the workers.
      - `-mca`: sets the MPI communication interface. Use the setting specified above to avoid possible multiprocessing and network communication issues.
      
          * The above configuration uses the `ens3` network interface. If this interface does not work in your environment (e.g., yielding error message `Unknown interfance name`), you may want to use a different interface (Run cmd `ifconfig` to see alternative interfaces in your environment.)
          
  * Language model configurations:
      - `--config`: specifies the config file to use. E.g., the above use the configuration defined in config_small.py
      - `--data_path`: specifies the directory containing PTB raw data (e.g., ptb.train.txt). If the data files do not exist, the program will automatically download, extract, and pre-process the data.

The model will begin training on the specified GPUs, and evaluate on the validation data periodically. Evaluation on the test data is performed after the training is done. Note that both validation and test are performed only on the rank-0 GPU (i.e., they are not distributed). 

## Results ##

We did simple test on two AWS p2.xlarge instances. 
Since the language model is small and the communication cost is considerable, as expected, the example here doesn't scale very well on 2-GPU 2-machine in terms of speedup rate. The perplexity results of multi-GPU are the same with those of single-GPU.

| config | epochs | train | valid  | test  | time/epoch (2-gpu) | time/epoch (single-gpu) |
| -------| -------| ------| -------| ------| -----| -----|
| small  | 13     | 40.81 | 118.99 | 114.72| 207s | 137s |
| medium | 39     | 44.18 |  87.63 |  84.42| 461s | 311s |
| large  | 55     | 36.54 |  82.55 |  78.72| 1765s | 931s |
