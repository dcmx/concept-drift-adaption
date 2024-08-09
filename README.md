# Concept Drift Adaption
Client-Side Adaptation to Concept Drift in Federated Learning

This repository contains the code to reproduce the figures of our paper "Client-Side Adaptation to Concept Drift in Federated Learning".

## Original Codebase

This project is based on Federated-Learning-in-Pytorch by Seok-Ju Hahn, which is licensed under the MIT License. You can find the original codebase [here](https://github.com/vaseline555/Federated-Learning-in-PyTorch).

## Requirements
* To guarantee compatibility use python version 3.10.10.
* The required libraries can be installed using `pip3 install -r requirements.txt`.
* To train using cuda please refer to the official [installation guide](https://pytorch.org/get-started/locally/).

## Run Experiments
* For general information on execution parameters use `python3 main.py -h`.
* To reproduce the experiments from our paper refer to the shell scripts provided in the [scripts directory](./scripts/).