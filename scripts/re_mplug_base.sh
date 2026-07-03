#!/bin/sh

python train.py \
    --config ./configs/re_mplug_base.yaml \
    --checkpoint ./mplug_base.pth \
    --output_dir output/re_mplug_base \
    --do_two_optim \
    --add_object \
    --max_input_length 80 \
    --max_length 50 \
    --add_ocr \
    --no_eval
