## Emacs, make this -*- mode: sh; -*-
 
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

MAINTAINER "Miguel Angel Davila Romero" madr0008@red.ujaen.es

## Update and Upgrade
RUN apt update
RUN apt upgrade

#Python
RUN apt install python3

#Cuda
RUN apt install nvidia-smi
RUN add-repository ppa:graphics-drivers/ppa
RUN add-repository ppa:graphics-drivers/ppa
RUN apt install nvidia-driver-455

#Conda
RUN apt install wget
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh
