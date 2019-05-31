#!/usr/bin/env bash
export PYTHONIOENCODING=utf-8
# preprocessing and train
nohup python -u main.py --config ./configs/word.yml -p --train --test > nohup.out 2>&1 &

# train only
# nohup python -u main.py --config ./configs/word.yml --train > nohup.out 2>&1 &

# test
# nohup python -u main.py --config ./configs/word.yml --test > nohup.out 2>&1 &

