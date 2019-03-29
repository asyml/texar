# Docker images for Texar

## Build the docker image

If you don't have Docker, please [install docker](https://docs.docker.com/engine/installation/) first.

This Dockerfile is supplied to build docker image with Ubuntu 16.04, cuda 10.0 and cudnn v7. The image uses Miniconda to configure python environment, ***Note that*** Tensorflow version is `1.13.0-rc0`. The Miniconda uses python3.6 by default, but you can pass `-e PYTHON_VERSION=x.y` flag in the building cmd to specify the python version for Miniconda, e.g. `-e PYTHON_VERSION=2.7` to use pytho2.7 in the image environment, otherwise leave it unset to use the default python3.6.

The image need to be built from `texar` repo directory, because docker needs to copy `texar` repo into its filesystem to install Texar into the image. So under current directory, run the following cmd to build image:

```bash
cd .. 
docker build -f docker/Dockerfile -t texar .
```

## Check Texar installation

To check if Texar is installed correctly, firstly run container shell from the image, use following cmd:

```bash
docker run -it texar /bin/bash
```

Then import Texar in python, we can run the following cmd into container shell:

```bash
python -c "import texar as tx;print(tx.data)"
```

If Texar is installed correctly, it should show:

```bash
<module 'texar.data' from '/opt/texar/texar/data/__init__.py'>
```

To exit the container shell, press `Ctrl` + `d`.

