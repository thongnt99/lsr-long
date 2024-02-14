[![DOI](https://zenodo.org/badge/625236818.svg)](https://zenodo.org/doi/10.5281/zenodo.10659505)

Code for SIGIR 2023 paper: Adapting Learned Sparse Retrieval to Long Documents

## Installation 
- Python packages
```console
conda create --name lsr python=3.9.12
conda activate lsr
pip install -r requirements.txt
```
- Anserini for inverted indexing & retrieval:  Clone and compile [anserini-lsr](https://github.com/thongnt99/anserini-lsr), a customized version of Anserini for learned sparse retrieval. When compiling, add ```-Dmaven.test.skip=true``` to skip the tests.

## Downloading and spliting data 

* MSMARCO Documents
```console
bash scripts/prepare_msmarco_doc.sh
```

* TREC-Robust04

```console
bash scripts/prepare_robust04.sh 
```
## BM25 baselines
```console
bash scripts/run_bm25_msmarco.sh 
bash scripts/run_bm25_robust04.sh 
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

* Evaluating on TREC Robust04 (zero-shot)
```bash 
bash scripts/evaluate_exact_sdm_trec_robust04.sh
```
### SoftSDM
* Estimating weights/Evaluating on MSMARCO Documents <br> Note: using ```+model.window_sizes=[1,2] +model.proximity=8``` generally leads to better performance on MSMARCO document but hurts TREC-Robust04 scores.


| #Passages | MRR@10 | R@1000 | Script | 
|--------------|--------|--------|---------|
| 1            |  36.98 | 95.49  | ```scripts/train_script_sdm_long_reranker_1_psg.sh``` |  
| 2            |  37.53 | 96.51  | ```scripts/train_script_sdm_long_reranker_2_psg.sh``` |  
| 3            |  37.41 | 96.76  | ```scripts/train_script_sdm_long_reranker_3_psg.sh``` |  
| 4            |  36.80 | 96.71  | ```scripts/train_script_sdm_long_reranker_4_psg.sh``` |  
| 5            |  36.79 | 96.61  | ```scripts/train_script_sdm_long_reranker_5_psg.sh``` |  

* Evaluating on TREC Robust04 (zero-shot)
```bash 
bash scripts/evaluate_sdm_trec_robust04.sh
```
## Citing and Authors 
If you find this repository helpful, please cite our following papers:
- Adapting Learned Sparse Retrieval for Long Documents
```bibtex
@inproceedings{nguyen:sigir2023-llsr,
  author = {Nguyen, Thong and MacAvaney, Sean and Yates, Andrew},
  title = {Adapting Learned Sparse Retrieval for Long Documents},
  booktitle = {Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year = {2023}
}

```
- A Unified Framework for Learned Sparse Retrieval
```bibtex
@inproceedings{nguyen2023unified,
  title={A Unified Framework for Learned Sparse Retrieval},
  author={Nguyen, Thong and MacAvaney, Sean and Yates, Andrew},
  booktitle={Advances in Information Retrieval: 45th European Conference on Information Retrieval, ECIR 2023, Dublin, Ireland, April 2--6, 2023, Proceedings, Part III},
  pages={101--116},
  year={2023},
  organization={Springer}
}
```


