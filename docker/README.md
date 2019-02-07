# Docker images for Texar

## Build the docker image

First [install docker](https://docs.docker.com/engine/installation/), and notice that [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) will be installed to run on Nvidia GPUs. Then in the main directory `#/texar` run the following command:

```bash
docker build -f docker/Dockerfile -t texar .
```

To run the image with its bash shell:

```bash
docker run -it texar /bin/bash
```

Then you can run Texar in python, e.g.:

```bash
#python -c "import texar as tx;print(tx.data)"
<module 'texar.data' from '/opt/texar/texar/data/__init__.py'>
```

## Hosted containers

The image is based on Ubuntu 16.04, with cuda 9.0 and cuDNN 7
