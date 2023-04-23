#!/bin/bash
echo "Exp results" > results_robust04_rm3.txt
num_psgs=(1 2 3 4 5 6 7 8 9 10)
for n in ${num_psgs[@]}; do 
mkdir data/trec-robust04/splits_psg_${n}
for f in data/trec-robust04/splits_psg/* ; do
    python lsr/preprocess/prepare_bm25.py $f $n & 
    pids="$pids $!"
done
wait $pids

../anserini-lsr/target/appassembler/bin/IndexCollection -collection JsonCollection -input data/trec-robust04/splits_psg_${n}/ -index indexes/trec-robust04-${n}-bm25 -generator DefaultLuceneDocumentGenerator -threads 60 -storePositions

../anserini-lsr/target/appassembler/bin/SearchCollection -index indexes/trec-robust04-${n}-bm25 -topics data/trec-robust04/desc-queries.tsv -topicreader TsvInt -output runs/trec-robust04_bm25_${n}.trec -parallelism 60 -bm25 -hits 1000 -bm25 -bm25.k1 0.9 -bm25.b 0.4 -rm3 -rm3.fbTerms 10 -rm3.fbDocs 10 -rm3.originalQueryWeight 0.5

echo $n >> results_robust04_rm3.txt
ir_measures data/trec-robust04/robust04.qrels runs/trec-robust04_bm25_${n}.trec NDCG@10 R@1000 >> results_robust04_rm3.txt

unset pids 
done