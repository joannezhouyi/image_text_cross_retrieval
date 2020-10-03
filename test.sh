#!/usr/bin/env bash

clear

case "$1" in

    0)
    echo "run start"
    CUDA_VISIBLE_DEVICES=0 python test.py --model_name='vse' --batch_size=128
    ;;

    *)
    echo
    echo "No input"
    ;;
esac


