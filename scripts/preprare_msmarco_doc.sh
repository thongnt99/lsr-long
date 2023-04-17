echo "Download msmarco documents"
python lsr/preprocess/prepare_msmarco_doc.py
# Split the collection into 60 splits 
echo "Split the collection into 60 partitions"
split --numeric-suffixes=1 --number=l/60  dât/msmarco_doc/collection.tsv data/msmarco_doc/splits/part
# Split long documents into passages 
echo "Split long documents into passages"
for f in data/msmarco_doc/splits/*;
do
echo $f
python lsr/long_documents/split_long_documnets.py $f & # remove the & if you want to run sequentially
done
echo "Running inference on document passages"
