#!/bin/bash
model=outputs/qmlp_dmlm_distil_sentence_transformer_kl_l1_0.0001/msmarco_doc
aggr="mean"
pids=""
echo "Exp results" > results.txt
num_psgs=(2 3 4 5 6 7 10 20)
for n in ${num_psgs[@]}; do 
rm -r ${model}/index_${aggr}
rm -r ${model}/doc_${aggr}
mkdir ${model}/doc_${aggr}
for f in $model/doc/*; do 
    python lsr/long_documents/aggregate_long_documents.py $f $aggr $n & 
    pids="$pids $!"
done

wait $pids

../anserini-lsr/target/appassembler/bin/IndexCollection -collection JsonTermWeightCollection -input ${model}/doc_${aggr}  -index ${model}/index_${aggr}  -generator TermWeightDocumentGenerator -threads 200 -impact -pretokenized

../anserini-lsr/target/appassembler/bin/SearchCollection -index ${model}/index_${aggr}  -topics ${model}/query/docdev.tsv  -topicreader TsvString -output ${model}/run_${aggr}_${n}.trec  -impact -pretokenized -hits 1000 -parallelism 200
echo $n >> results.txt
ir_measures data/msmarco_doc/msmarco-docdev-qrels.tsv ${model}/run_${aggr}_${n}.trec MRR@10 R@1000 >> results.txt

unset pids 
done