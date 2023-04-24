#!/bin/bash
input_dir=data/trec-robust04
aggr="mean"
pids=""
output_file=${input_dir}/results_${aggr}.txt
echo "Exp results" > $output_file
num_psgs=(1 2 3 4 5 6 7 10)
for n in ${num_psgs[@]}; do 
rm -r ${input_dir}/index_${aggr}
rm -r ${input_dir}/doc_vectors_${aggr}
mkdir ${input_dir}/doc_vectors_${aggr}
for f in $input_dir/doc_vectors/*; do 
    python lsr/long_documents/aggregate_long_documents.py $f $aggr $n & 
    pids="$pids $!"
done
wait $pids

../anserini-lsr/target/appassembler/bin/IndexCollection -collection JsonSparseVectorCollection -input ${input_dir}/doc_vectors_${aggr}  -index ${input_dir}/index_${aggr}  -generator SparseVectorDocumentGenerator -threads 200 -impact -pretokenized

../anserini-lsr/target/appassembler/bin/SearchCollection -index ${input_dir}/index_${aggr}  -topics ${input_dir}/query.tsv -topicreader TsvString -output ${input_dir}/run_${aggr}_${n}.trec  -impact -pretokenized -hits 1000 -parallelism 200

echo $n >> $output_file

ir_measures data/msmarco_doc/msmarco-docdev-qrels.tsv ${input_dir}/run_${aggr}_${n}.trec MRR@10 NDCG@10 R@1000 >> $output_file

unset pids 
done