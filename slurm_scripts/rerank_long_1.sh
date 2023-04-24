#!/bin/bash
#SBATCH --nodes 1
#SBATCH -p gpu20
#SBATCH --gres gpu:1
#SBATCH -c 5
#SBATCH --mem-per-cpu 15000
#SBATCH -t 0-03:00:00
#SBATCH -o slurm_scripts/logs/rerank_long_1.out
#SBATCH -e slurm_scripts/logs/rerank_long_1.err
# Make conda available:
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate lsr 
python -m lsr.rerank_long -run data/msmarco_doc/run_max_score_1.trec -q data/msmarco_doc/msmarco-docdev-queries.tsv -d data/msmarco_doc/collection_psgs.tsv -npsg 1 -cp outputs/reranker_exact_qmlp_dmlm_msmarco_doc_ce_1_psg/checkpoint-20000 -qrel data/msmarco_doc/msmarco-docdev-queries.tsv