#!/bin/bash
#SBATCH -A danielk_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --output=111.out
#SBATCH --time=72:00:00
module load anaconda
conda info --envs
source activate big


user_id=lshen30
#proj_dir=/data/danielk/${user_id}/instruction
#export PYTHONPATH="/data/danielk/${user_id}/flat"
#export CUDA_VISIBLE_DEVICES=0,1,2,3
deepspeed --num_gpus=1  run_clm.py \
--deepspeed ds_config_gptneo.json \
--model_name_or_path EleutherAI/gpt-neo-2.7B \
--train_file train.csv \
--validation_file validation.csv \
--do_train \
--do_eval \
--fp16 \
--overwrite_cache \
--evaluation_strategy="epoch" \
--output_dir finetuned \
--eval_steps 2 \
--num_train_epochs 2 \
--gradient_accumulation_steps 2 \
--per_device_train_batch_size 2