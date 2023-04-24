Code for SIGIR 2023 paper: Adapting Learned Sparse Retrieval to Long Documents
## Installation 
```console
conda create --name lsr python=3.9.12
conda activate lsr
pip install -r requirements.txt
```
## Downloading and spliting data 

* MSMARCO Documents
```console
bash scripts/prepare_msmarco_doc.sh
```

* TREC-Robust04

```console
bash scripts/prepare_robust04.sh 
```

## Simple aggregation 
To perform aggregation on MSMARCO, follow these steps. For TREC-Robust04, please modify the input and output files accordingly.
#### 1. Running inferences on segments (passages) and queries:
- segment inference (can be distributed on multiple gpus to speed up)
```console
for i in {1..60}
do
input_path=data/msmarco_doc/splits_psg/part$(printf "%02d" $i)
output_path=data/msmarco_doc/vectors/part$(printf "%02d" $i)
batch_size=256
type='doc'
python -m lsr.inference --inp $input_path --out $output_path --type $type --bs $batch_size
done
```
- query inference
```console
input_path=data/msmarco_doc/msmarco-docdev-queries.tsv
output_path=data/msmarco_doc/query.tsv
batch_size=256
type='query'
python -m lsr.inference --inp $input_path --out $output_path --type $type --bs $batch_size
```
#### 2. Aggregating
- Representation aggregation
```console
bash scripts/aggregate_rep_msmarco_doc.sh 
```
- Score (max) aggregation
```console
bash scripts/aggregate_score_msmarco_doc.sh
``` 
## ExactSDM and SoftSDM

### ExactSDM 
* Estimating weights/Evaluating on MSMARCO Documents 

| #Passages | MRR@10 | R@1000 | Script | 
|--------------|--------|--------|---------|
| 1            |  37.08 | 95.49  | ```scripts/train_script_exact_sdm_long_reranker_1_psg.sh``` |  
| 2            |  37.45 | 96.51  | ```scripts/train_script_exact_sdm_long_reranker_2_psg.sh``` |  
| 3            |  37.36 | 96.76  | ```scripts/train_script_exact_sdm_long_reranker_3_psg.sh``` |  
| 4            |  37.03 | 96.71  | ```scripts/train_script_exact_sdm_long_reranker_4_psg.sh``` |  
| 5            |  36.95 | 96.61  | ```scripts/train_script_exact_sdm_long_reranker_5_psg.sh``` |  

* Evaluating on TREC Robust04
### SoftSDM
* Evaluating on MSMARCO Documents 
* Evaluating on TREC Robust04


