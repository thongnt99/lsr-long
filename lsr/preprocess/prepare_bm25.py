from collections import defaultdict
import json

inp_path = "data/msmarco_doc/collection_psgs.tsv"
out_path = "data/msmarco_doc/collection_psgs_1.tsv"
top_k = 1
doc2psgs = defaultdict(list)
with open(inp_path, "r") as f:
    for line in f:
        doc_psg, text = line.strip().split("\t")
        doc_id, psg_id = doc_psg.split("@@")
        doc2psgs[doc_id].append(text)

with open(out_path, "w") as f:
    for doc_id in doc2psgs:
        psgs = doc2psgs[doc_id][:top_k]
        text = " ".join(psgs)
        json_doc = {"id": doc_id, "text": text}
        f.write(json.dumps(json_doc)+"\n")
