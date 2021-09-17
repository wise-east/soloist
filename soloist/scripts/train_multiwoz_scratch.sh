#!/bin/bash
#SBATCH --job-name=scratch-soloist
#SBATCH --partition=a100 
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00 # run for one day
#SBATCH --cpus-per-task=10
## %j is the job id, %u is the user id
#SBATCH --output=/data/home/justincho/project/soloist/soloist/scripts/slurm_logs/finetune_scratch-%j.log

source ~/.bashrc
source /data/home/justincho/miniconda/etc/profile.d/conda.sh
conda activate soloist

cd /data/home/justincho/project/soloist/soloist


# lr 1e-5 to 5e-5
# mc_loss_efficient 0.1 to 1
# etc.
N_EPOCH=20

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
--nproc_per_node=8 \
--nnodes=1 \
--node_rank=0 \
--master_addr="localhost" \
--master_port=8898 soloist_train.py \
--output_dir="multiwoz_scratch_models_${N_EPOCH}" \
--model_type=gpt2 \
--model_name_or_path=gpt2 \
--do_eval \
--eval_all_checkpoints \
--train_data_file=../examples/multiwoz/train.soloist.json  \
--eval_data_file=../examples/multiwoz/valid.soloist.json  \
--add_special_action_tokens=../examples/multiwoz/resource/special_tokens.txt \
--per_gpu_train_batch_size 6 \
--per_gpu_train_batch_size 12 \
--num_train_epochs $N_EPOCH \
--learning_rate 5e-5 \
--overwrite_cache \
--save_steps 5000 \
--max_seq 512 \
--overwrite_output_dir \
--max_turn 15 \
--num_candidates 1 \
--mc_loss_efficient 0.33 \
--add_response_prediction \
--add_same_belief_response_prediction \
--add_belief_prediction \
# --do_train \
