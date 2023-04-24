echo "Download msmarco documents"
python lsr/preprocess/prepare_msmarco_doc.py
# Split the collection into 60 splits 
echo "Split the collection into 60 partitions"
mkdir -p data/msmarco_doc/splits
split --numeric-suffixes=1 --number=l/60  data/msmarco_doc/collection.tsv data/msmarco_doc/splits/part
# Split long documents into passages 
echo "Split long documents into passages"
mkdir -p data/msmarco_doc/splits_psg
pids=""
for f in data/msmarco_doc/splits/*;
do
echo $f;
python lsr/long_documents/split_long_documents.py $f & # remove the & if you want to run sequentially
pids="$pids $!"
done
wait $pids
cat data/msmarco_doc/splits/* > data/msmarco_doc/collection_psgs.tsv
unset pids 