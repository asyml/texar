# Distributed Version of Language Model on PTB

This is the distributed version of language model on PTB with Horovod framework.

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

- Basic setting of Horovod in your codes. Generally, you will need in insert the horovod wrappers in appropriate portion of your codes
    - `hvd.init()`: initialize the horovod
    - set visible GPU list by `config.gpu_options.visible_device_list = str(hvd.local_rank())`, to make each process see the attached single GPU.
    - `hvd.DistributedOptimizer`: wrap your optimizer.
    - `hvd.broadcast_global_variables(0).run()` broadcast your global variables to different processes from rank-0 process.
- Data feeding:
    - You should split your dataset into shards before sending them to different processes, to make sure different GPUs are fed different mini-batch in each iteration.
    - Because we update the global variables based on the mini-batches in different processes, we may need to adjust the `learning rate`, `batch_size` to fit the distributed settings. In this example, we scale down the specified `batch_size` with the number of processes before feeding the mini-batch into the graph, to replicate the gradient computation in single-gpu setting.

## Usage ##

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

Here:
  * `-np`: number of processes
  * `-H`: specifies the address of different servers and the number of processes used in each server.
  * `--bind-to none`: specifies Open MPI to not bind a training process to a single CPU core (which would hurt performance).
  * `-map-by slot`: allows you to have a mixture of different NUMA configurations because the default behavior is to bind to the socket.
  * `-mca`: 
  * `-x`: to specify (-x NCCL_DEBUG=INFO) or copy (-x LD_LIBRARY_PATH) an environment variable to all the workers.

The model will begin training, and will evaluate on the validation data periodically, and evaluate on the test data after the training is done. 


## Results ##

As per the TensorFlow official PTB example, the perplexity of different configs is:

| config | epochs | train | valid  | test  | time/epoch (2-gpu) | time/epoch(single-gpu) |
| -------| -------| ------| -------| ------|
| small  | 13     | 40.81 | 118.99 | 114.63|    | 137.004802s
| medium | 39     | 44.18 |  87.63 |  84.42|
| large  | 55     | 37.87 |  82.62 |  78.29|
