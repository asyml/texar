# Docker images for Texar

## Build the docker image

If you don't have Docker, please [install docker](https://docs.docker.com/engine/installation/) first.

This Dockerfile is supplied to build docker image with Ubuntu 16.04, cuda 10.0 and cudnn v7. The image uses Miniconda to configure python environment, ***Note that*** Tensorflow version is `1.13.0` in the environment. 

The image need to be built from `texar` repo directory, because docker needs to copy `texar` repo into its filesystem to install Texar into the image. So under current directory, run the following cmds to build image:

```bash
cd .. 
docker build -f docker/Dockerfile -t texar .
```

The Miniconda uses python3.6 by default, but you can pass `--build-arg PYTHON_VERSION=x.y` flag in the building cmd to specify the python version for Miniconda. E.g. under current directory, run following cmds to build image with python2.7:

```bash
cd ..
docker build --build-arg PYTHON_VERSION=2.7 -f docker/Dockerfile  -t texar  . 
```

Otherwise leave it unset to use the default python3.6

## Check Texar installation

To check if Texar is installed correctly, firstly run container shell from the image, use following cmd:

```bash
docker run -it texar /bin/bash
```

Then import Texar in python, we can run the following cmd into container shell:

```bash
python -c "import texar as tx;print(tx.data)"
```

If Texar is installed correctly, the shell should return:

```bash
<module 'texar.data' from '/opt/texar/texar/data/__init__.py'>
```

To exit the container shell, press `Ctrl` + `d`.
