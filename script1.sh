#!/bin/zsh
max=100
for i in `seq 1 $max`
do
    python main.py 100 $i 1
done