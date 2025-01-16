#!/bin/bash

python -m hw2.experiments run-exp --run-name exp1_4 -K 64 128 256 -L 2 --bs-train 100 --bs-test 24 --batches 100 --epochs 100 --early-stopping 3 --lr 0.001 --reg 0.001 --pool-every 4 --hidden-dims 100 --model-type resnet --device mps
python -m hw2.experiments run-exp --run-name exp1_4 -K 64 128 256 -L 4 --bs-train 100 --bs-test 24 --batches 100 --epochs 100 --early-stopping 3 --lr 0.001 --reg 0.001 --pool-every 4 --hidden-dims 100 --model-type resnet --device mps
python -m hw2.experiments run-exp --run-name exp1_4 -K 64 128 256 -L 8 --bs-train 100 --bs-test 24 --batches 100 --epochs 100 --early-stopping 3 --lr 0.001 --reg 0.001 --pool-every 4 --hidden-dims 100 --model-type resnet --device mps

python -m hw2.experiments run-exp --run-name exp1_2 -K 32 -L 2 --bs-train 100 --bs-test 24 --batches 100 --epochs 100 --early-stopping 3 --lr 0.001 --reg 0.001 --pool-every 4 --hidden-dims 100 --device mps
python -m hw2.experiments run-exp --run-name exp1_2 -K 64 -L 2 --bs-train 100 --bs-test 24 --batches 100 --epochs 100 --early-stopping 3 --lr 0.001 --reg 0.001 --pool-every 4 --hidden-dims 100 --device mps
python -m hw2.experiments run-exp --run-name exp1_2 -K 128 -L 2 --bs-train 100 --bs-test 24 --batches 100 --epochs 100 --early-stopping 3 --lr 0.001 --reg 0.001 --pool-every 4 --hidden-dims 100 --device mps

python -m hw2.experiments run-exp --run-name exp1_2 -K 32 -L 4 --bs-train 100 --bs-test 24 --batches 100 --epochs 100 --early-stopping 3 --lr 0.001 --reg 0.001 --pool-every 4 --hidden-dims 100 --device mps
python -m hw2.experiments run-exp --run-name exp1_2 -K 64 -L 4 --bs-train 100 --bs-test 24 --batches 100 --epochs 100 --early-stopping 3 --lr 0.001 --reg 0.001 --pool-every 4 --hidden-dims 100 --device mps
python -m hw2.experiments run-exp --run-name exp1_2 -K 128 -L 4 --bs-train 100 --bs-test 24 --batches 100 --epochs 100 --early-stopping 3 --lr 0.001 --reg 0.001 --pool-every 4 --hidden-dims 100 --device mps

python -m hw2.experiments run-exp --run-name exp1_2 -K 32 -L 8 --bs-train 100 --bs-test 24 --batches 100 --epochs 100 --early-stopping 3 --lr 0.001 --reg 0.001 --pool-every 4 --hidden-dims 100 --device mps
python -m hw2.experiments run-exp --run-name exp1_2 -K 64 -L 8 --bs-train 100 --bs-test 24 --batches 100 --epochs 100 --early-stopping 3 --lr 0.001 --reg 0.001 --pool-every 4 --hidden-dims 100 --device mps
python -m hw2.experiments run-exp --run-name exp1_2 -K 128 -L 8 --bs-train 100 --bs-test 24 --batches 100 --epochs 100 --early-stopping 3 --lr 0.001 --reg 0.001 --pool-every 4 --hidden-dims 100 --device mps




