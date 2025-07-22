#!/bin/bash

for C in {1..15}; do
    cmd="python pretrain_s1.py --test-subject $C --save-dir pretrain_s1"
    echo "Running: $cmd"
    $cmd
    cmd="python pretrain_s2.py --test-subject $C --save-dir pretrain_s2"
    echo "Running: $cmd"
    $cmd
done

# for C in {1..15}; do
#     for B in 11 12 13 14 15 21 22 23 24 25 31; do
#         cmd="python coldStart.py --pre-bb traditional --pre-cls traditional_maml --test-subject $C --save-dir coldStart_maml/$B"
#         echo "Running: $cmd"
#         $cmd
#     done
# done