#!/bin/bash
#SBATCH --job-name=decode_soloist
#SBATCH --partition=a100 
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00 # run for one day
#SBATCH --cpus-per-task=10
## %j is the job id, %u is the user id
#SBATCH --output=/data/home/justincho/project/soloist/soloist/scripts/slurm_logs/decode_soloist-%j.log

source /data/home/justincho/miniconda/etc/profile.d/conda.sh
conda activate soloist

cd /data/home/justincho/project/soloist/soloist/

# temp 0.7 - 1.5
# top_p 0.2 - 0.8
# CHECKPOINT saved checkpints, valid around 40k to 80k
NS=1
TEMP=1
TOP_P=0.5
BATCHSIZE=16
# VERSION="scratch"
VERSION="prefinetuned" 
CHECKPOINT="multiwoz_${VERSION}_models_20/checkpoint-10000"
OUTDIR="${CHECKPOINT}/decoded_testset_${VERSION}.json"

echo $BATCHSIZE
echo $NS
echo $VERSION
echo $CHECKPOINT
echo $OUTDIR

python soloist_decode.py \
--model_type=gpt2 \
--model_name_or_path=$CHECKPOINT \
--num_samples $NS \
--input_file ../examples/multiwoz/test.soloist.json \
--top_p $TOP_P \
--temperature $TEMP \
--output_file $OUTDIR \
--max_turn 15 \
--batch_size $BATCHSIZE
# --do_batch \
# --batch_test
