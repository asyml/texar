FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
ARG PYTHON_VERSION=3.6
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         curl \
         ca-certificates \
     && rm -rf /var/lib/apt/lists/*


RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION  && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

WORKDIR /opt/texar
COPY . .
RUN pip install -e . && pip install -r requirements.txt
RUN pip install tensorflow==1.13.0-rc0
RUN pip install tensorflow-gpu==1.13.0-rc0
ENV LD_LIBRARY_PATH /usr/local/cuda-10.0/compat:$LD_LIBRARY_PATH


WORKDIR /workspace
RUN chmod -R a+w /workspace

