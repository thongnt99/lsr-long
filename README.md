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


