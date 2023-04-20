#!/bin/bash
#SBATCH --nodes 1
#SBATCH -p d5gpu
#SBATCH --gres gpu:1
#SBATCH -t 00-10:00:00
#SBATCH -o slurm_scripts/logs/sdm_long_reranker_2_psg.out
#SBATCH -e slurm_scripts/logs/sdm_long_reranker_2_psg.err
# Make conda available:
eval "$(conda shell.bash hook)"
# Activate a conda environment:

conda activate lsr
python -m lsr.train +experiment=reranker_qmlp_dmlm_msmarco_doc_2_psg resume_from_checkpoint=lsr42/qmlp_dmlm_msmarco_distil_kl_l1_0.0001 training_arguments.fp16=True training_arguments.per_device_train_batch_size=8 +training_arguments.learning_rate=0.001 training_arguments.max_steps=60000 +model.window_sizes=[1,2] +model.proximity=8 training_arguments.evaluation_strategy="steps" +training_arguments.eval_steps=20000 training_arguments.save_steps=20000 +training_arguments.save_total_limit=2 +training_arguments.metric_for_best_model="RR@10" +training_arguments.per_device_eval_batch_size=16