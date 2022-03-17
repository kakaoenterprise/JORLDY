FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

RUN apt-get update \
    && apt-get -y install libgl1-mesa-glx \
    && apt-get -y install libglib2.0-0 \
    && apt-get -y install git \
    && apt-get -y install gifsicle

# mujoco
RUN apt-get -y install wget \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz \
    && mkdir ~/.mujoco \
    && mv mujoco210-linux-x86_64.tar.gz ~/.mujoco/ \
    && tar -xvzf ~/.mujoco/mujoco210-linux-x86_64.tar.gz -C ~/.mujoco/ \
    && apt-get -y install libosmesa6-dev libglfw3 libgl1-mesa-dev \
    && cp -r ~/.mujoco/mujoco210/bin/* /usr/lib/ \
    && pip install 'mujoco-py<2.2,>=2.1'
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin

COPY requirements.txt /
RUN pip install -r /requirements.txt

WORKDIR /JORLDY
