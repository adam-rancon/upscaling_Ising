# Upscaling the critical Ising model

*started December 2022*

## Introduction
This repository contains the code used to generate the data of the manuscript "Dreaming up scale invariance via inverse renormalization group"

## Installation

First create a Python environment with conda: `conda create -n ising python=3.8`.

Install required libraries with: `pip3 install -r requirements.txt`.


## Training data

An example of training data is given for the 2D Ising model at criticality of size 8x8.
There are 10000 configurations for training and 1000 for validation.

The training of the simplest model is given in training_models.ipynb

The data is in a highly compressed format, which can be read with the class MyFantasticDataset,
 which items are tensors of 0 and 1 of of shape 8x8.
 
 In practice, data need to be tensors of shape N_training x L x L.
 
 We also show how to train the simplest model of kernel size 3, and how we upscale from a system of size 1x1




## Models parameters trained with the weighted loss are in the folder "weighted_loss_models", though without in "models"

The naming convention is :
	upsamplerconv_32 is the more complicated model with a non-linear layer in the middle
	upsampler3par is the simplest with 3 parameters (and so on for 6par and 10par)
	k and ks are kernel sizes
	learning_r is the learning rate
	L is the system size of the training data
	batch is the batch size use while training
	set_size the number of training data
	
	
## Data analysis 

It is performed in calculate_observables.py which uses observables_v.py
