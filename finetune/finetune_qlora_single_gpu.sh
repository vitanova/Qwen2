#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

MODEL="Qwen/Qwen-7B-Chat-Int4" # Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="finetune/data2.json"

function usage() {
    echo '
Usage: bash finetune/finetune_qlora_single_gpu.sh [-m MODEL_PATH] [-d DATA_PATH]
'
}

while [[ "$1" != "" ]]; do
    case $1 in
        -m | --model )
            shift
            MODEL=$1
            ;;
        -d | --data )
            shift
            DATA=$1
            ;;
        -h | --help )
            usage
            exit 0
            ;;
        * )
            echo "Unknown argument ${1}"
            exit 1
            ;;
    esac
    shift
done

export CUDA_VISIBLE_DEVICES=0

# Remember to use --fp16 instead of --bf16 due to autogptq
python finetune.py \
  --model_name_or_path $MODEL \
  --data_path $DATA \
  --fp16 True \
  --output_dir output_qwen \
  --num_train_epochs 1 \
  --per_device_train_batch_size 6 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 100 \
  --save_total_limit 1 \
  --learning_rate 4e-4 \
  --weight_decay 0.1 \
  --adam_beta2 0.95 \
  --warmup_ratio 0.2 \
  --lr_scheduler_type "constant_with_warmup" \
  --logging_steps 1 \
  --report_to "none" \
  --model_max_length 256 \
  --lazy_preprocess True \
  --gradient_checkpointing \
  --use_lora \
  --q_lora \
  --deepspeed finetune/ds_config_zero2.json
