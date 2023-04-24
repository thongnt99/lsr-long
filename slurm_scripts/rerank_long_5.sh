#!/bin/bash
#SBATCH --nodes 1
#SBATCH -p gpu20
#SBATCH --gres gpu:1
#SBATCH -c 5
#SBATCH --mem-per-cpu 15000
#SBATCH -t 0-03:00:00
#SBATCH -o slurm_scripts/logs/rerank_long_5.out
#SBATCH -e slurm_scripts/logs/rerank_long_5.err
conda activate lsr 
python -m lsr.rerank_long -run data/msmarco_doc/run_max_score_5.trec -q data/msmarco_doc/msmarco-docdev-queries.tsv -d data/msmarco_doc/collection_psgs.tsv -npsg 5 -cp outputs/reranker_exact_qmlp_dmlm_msmarco_doc_ce_5_psg/checkpoint-20000