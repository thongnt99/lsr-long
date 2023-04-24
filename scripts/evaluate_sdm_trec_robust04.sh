
conda activate lsr 

echo "using 1 segment"
python -m lsr.rerank_long -run data/trec-robust04/run_max_score_1.trec -q data/trec-robust04/desc-queries.tsv -d data/trec-robust04/collection_psgs.tsv -npsg 1 -cp outputs/reranker_qmlp_dmlm_msmarco_doc_ce_1_psg/checkpoint-20000 -qrel data/trec-robust04/robust04.qrels -m soft 

echo "using 2 segments"
python -m lsr.rerank_long -run data/trec-robust04/run_max_score_2.trec -q data/trec-robust04/desc-queries.tsv -d data/trec-robust04/collection_psgs.tsv -npsg 2 -cp outputs/reranker_qmlp_dmlm_msmarco_doc_ce_2_psg/checkpoint-20000 -qrel data/trec-robust04/robust04.qrels -m soft 

echo "using 3 segments"
python -m lsr.rerank_long -run data/trec-robust04/run_max_score_3.trec -q data/trec-robust04/desc-queries.tsv -d data/trec-robust04/collection_psgs.tsv -npsg 3 -cp outputs/reranker_qmlp_dmlm_msmarco_doc_ce_3_psg/checkpoint-20000 -qrel data/trec-robust04/robust04.qrels -m soft 

echo "using 4 segments"
python -m lsr.rerank_long -run data/trec-robust04/run_max_score_4.trec -q data/trec-robust04/desc-queries.tsv -d data/trec-robust04/collection_psgs.tsv -npsg 4 -cp outputs/reranker_qmlp_dmlm_msmarco_doc_ce_4_psg/checkpoint-20000 -qrel data/trec-robust04/robust04.qrels -m soft 

echo "using 5 segments"
python -m lsr.rerank_long -run data/trec-robust04/run_max_score_5.trec -q data/trec-robust04/desc-queries.tsv -d data/trec-robust04/collection_psgs.tsv -npsg 5 -cp outputs/reranker_qmlp_dmlm_msmarco_doc_ce_5_psg/checkpoint-20000 -qrel data/trec-robust04/robust04.qrels -m soft 
