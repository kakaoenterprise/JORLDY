FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

RUN apt-get update 
RUN apt-get -y install libgl1-mesa-glx
RUN apt-get -y install libglib2.0-0
RUN apt-get -y install git
RUN apt-get -y install gifsicle

# mujoco
RUN apt-get -y install wget
RUN wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz 
RUN mkdir ~/.mujoco
RUN mv mujoco210-linux-x86_64.tar.gz ~/.mujoco/
RUN tar -xvzf ~/.mujoco/mujoco210-linux-x86_64.tar.gz -C ~/.mujoco/
RUN apt-get -y install libosmesa6-dev libglfw3 libgl1-mesa-dev
RUN cp -r ~/.mujoco/mujoco210/bin/* /usr/lib/
RUN pip install 'mujoco-py<2.2,>=2.1'
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin

COPY requirements.txt /
RUN pip install -r /requirements.txt

WORKDIR /JORLDY
