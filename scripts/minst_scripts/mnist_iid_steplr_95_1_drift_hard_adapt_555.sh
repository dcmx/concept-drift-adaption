#!/bin/sh
python3 main.py --use_tb --drift_adaptation --b1 0.5 --b2 0.5 --b3 0.5 --concept_drift --drift_start 40 --drift_duration 20 --drift_mode hard --log_path "./log/MNIST" --result_path "./result/MNIST" --exp_name MNIST_STEPLR_95_1_DRIFT_HARD_ADAPTATION_555 --seed 42 --device cuda --dataset MNIST --split_type iid --test_fraction 0 --model_name TwoCNN --resize 28 --hidden_size 200 --algorithm fedavg --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 precision recall --K 100 --R 100 --E 3 --C 0.1 --B 10 --beta 0 --optimizer SGD --lr 0.2 --lr_decay 0.95 --lr_decay_step 1 --criterion CrossEntropyLoss
