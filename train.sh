#!/usr/bin/env bash

clear

case "$1" in

    0)
    echo "run start"
    CUDA_VISIBLE_DEVICES=0 python train.py --model_name='nlp' --rnn_type='GRU' --text_enc_init 1 --batch_size=128 --data_path "data" --data_name 'coco_precomp' --logger_name 'runs/gru_cross' --max_violation
    ;;

    1)
    echo "run start"
    CUDA_VISIBLE_DEVICES=0 python train.py --model_name='nlp' --rnn_type='BiGRU' --text_enc_init 0 --batch_size=128 --data_path "data" --data_name 'coco_precomp' --logger_name 'runs/bigru_cross' --max_violation
    ;;

    *)
    echo
    echo "No input"
    ;;
esac


