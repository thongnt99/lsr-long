#!/bin/bash
echo "Exp results" > results.txt
num_psgs=(7 8 9 10)
for n in ${num_psgs[@]}; do 
mkdir data/msmarco_doc/splits_psg_${n}
for f in data/msmarco_doc/splits_psg/* ; do
    python lsr/preprocess/prepare_bm25.py $f $n & 
    pids="$pids $!"
done
wait $pids

../anserini-lsr/target/appassembler/bin/IndexCollection -collection JsonCollection -input data/msmarco_doc/splits_psg_${n}/ -index indexes/msmarco-doc-${n}-bm25 -generator DefaultLuceneDocumentGenerator -threads 60 -storePositions

../anserini-lsr/target/appassembler/bin/SearchCollection -index indexes/msmarco-doc-${n}-bm25 -topics data/msmarco_doc/msmarco-docdev-queries.tsv -topicreader TsvInt -output runs/msmarco_doc_bm25_${n}.trec -parallelism 60 -bm25 -hits 1000 -bm25

echo $n >> results.txt
ir_measures data/msmarco_doc/msmarco-docdev-qrels.tsv runs/msmarco_doc_bm25_${n}.trec MRR@10 R@1000 >> results.txt

unset pids 
done