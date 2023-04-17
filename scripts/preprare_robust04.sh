echo "Download msmarco documents"
python lsr/preprocess/prepare_robust04.py
# Split the collection into 60 splits 
echo "Split the collection into 60 partitions"
split --numeric-suffixes=1 --number=l/60 data/trec-robust04/collection.tsv data/trec-robust04/splits/part
# Split long documents into passages 
echo "Split long documents into passages"
for f in data/trec-robust04/splits/*;
do
echo $f
python lsr/long_documents/split_long_documnets.py $f & # remove the & if you want to run sequentially
done
echo "Running inference on document passages"