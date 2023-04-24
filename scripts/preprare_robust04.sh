echo "Download msmarco documents"
python lsr/preprocess/prepare_robust04.py
# Split the collection into 60 splits 
echo "Split the collection into 60 partitions"
mkdir -p data/trec-robust04/splits
split --numeric-suffixes=1 --number=l/60 data/trec-robust04/collection.tsv data/trec-robust04/splits/part
# Split long documents into passages 
echo "Split long documents into passages"
mkdir -p data/trec-robust04/splits_psg
pids=""
for f in data/trec-robust04/splits/*;
do
echo $f;
python lsr/long_documents/split_long_documents.py $f & # remove the & if you want to run sequentially
pids="$pid $!"
done
wait $pids 
cat data/trec-robust04/splits/* > data/trec-robust04/collection_psgs.tsv