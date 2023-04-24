#!/bin/bash
input_dir=data/msmarco_doc
output_file=${input_dir}/results_max_score.txt
echo "Indexing passages"

../anserini-lsr/target/appassembler/bin/IndexCollection -collection JsonSparseVectorCollection -input ${input_dir}/doc_vectors -index ${input_dir}/index_psgs  -generator SparseVectorDocumentGenerator -threads 200 -impact -pretokenized

echo "Retrieval top relevant passages"

../anserini-lsr/target/appassembler/bin/SearchCollection -index ${input_dir}/index_psgs  -topics ${input_dir}/query.tsv -topicreader TsvString -output ${input_dir}/run_psgs.trec  -impact -pretokenized -hits 10000 -parallelism 200

echo "Aggregating and evaluting"

echo "Exp results" > $output_file
num_psgs=(1 2 3 4 5 6 7 10)
for n in ${num_psgs[@]}; do 
echo $n >> $output_file
python lsr/long_documents/max_score_aggregation.py ${input_dir}/run_psgs.trec ${input_dir}/run_max_score_${n}.trec $n 
ir_measures data/msmarco_doc/msmarco-docdev-qrels.tsv ${input_dir}/run_max_score_${n}.trec MRR@10 NDCG@10 R@1000 >> $output_file 
done