FROM ubuntu:20.04
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*
COPY conda.yaml /opt/tmp/

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-py39_4.12.0-Linux-x86_64.sh -b \
    && rm -f Miniconda3-py39_4.12.0-Linux-x86_64.sh 

WORKDIR /home/prefect

RUN echo 'export PATH=/root/miniconda3/bin/:$PATH' >> /root/.bashrc 

RUN conda init bash && conda env  update -f /opt/tmp/conda.yaml  && \
    apt clean && \
    apt autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/*

ENV CONDA_DEFAULT_ENV vdo
COPY . /home/prefect

ENTRYPOINT ["/bin/bash"]
