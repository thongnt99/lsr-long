#!/bin/bash
#SBATCH --nodes 1
#SBATCH -p gpu20
#SBATCH --gres gpu:1
#SBATCH -c 5
#SBATCH --mem-per-cpu 15000
#SBATCH -t 0-01:00:00
#SBATCH -o slurm_scripts/logs/inference_robust04.out
#SBATCH -e slurm_scripts/logs/inference_robust04.err
#SBATCH -a 1-60%60
# Make conda available:
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate lsr 
trap "trap ' ' TERM INT; kill -TERM 0; wait" TERM INT
input_path=data/trec-robust04/splits_psg/part$(printf "%02d" $SLURM_ARRAY_TASK_ID)
output_path=data/trec-robust04/doc_vectors/part$(printf "%02d" $SLURM_ARRAY_TASK_ID)
batch_size=256
type='doc'
python -m lsr.inference --inp $input_path --out $output_path --type $type --bs $batch_size
wait
