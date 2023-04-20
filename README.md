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
Running inferences on segments (passages)

Aggregating

Indexing and evaluating

## ExactSDM and SoftSDM

### ExactSDM 
* Estimating weights/Evaluating on MSMARCO Documents 

| #Passages | MRR@10 | R@1000 | Script | 
|--------------|--------|--------|---------|
| 1            |  37.00 | 94.94  | ```scripts/train_script_exact_sdm_long_reranker_1_psg.sh``` |  
| 2            |  37.27 | 95.88  | ```scripts/train_script_exact_sdm_long_reranker_2_psg.sh``` |  
| 3            |  37.31 | 96.11  | ```scripts/train_script_exact_sdm_long_reranker_3_psg.sh``` |  
| 4            |  37.03 | 96.15  | ```scripts/train_script_exact_sdm_long_reranker_4_psg.sh``` |  
| 5            |  36.85 | 96.15  | ```scripts/train_script_exact_sdm_long_reranker_5_psg.sh``` |  

* Evaluating on TREC Robust04
### SoftSDM
* Evaluating on MSMARCO Documents 
* Evaluating on TREC Robust04


