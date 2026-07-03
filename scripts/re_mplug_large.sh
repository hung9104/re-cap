#!/bin/sh

python train.py \
    --config ./configs/re_mplug_large.yaml \
    --checkpoint ./mplug_large_v2.pth \
    --output_dir output/re_mplug_large \
    --do_two_optim \
    --add_object \
    --max_input_length 80 \
    --add_ocr \
    --no_eval
