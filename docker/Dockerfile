FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-devel

RUN apt-get update 
RUN apt-get -y install libgl1-mesa-glx
RUN apt-get -y install libglib2.0-0
RUN apt-get install gifsicle

COPY requirements.txt /
RUN pip install -r /requirements.txt
WORKDIR /jorldy
