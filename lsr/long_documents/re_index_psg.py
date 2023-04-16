import json
import sys
from collections import defaultdict
from tqdm import tqdm

doc_map = defaultdict(dict)

inp_path = sys.argv[1]
out_path = inp_path.replace("/doc", "/doc_re")
with open(inp_path, "r", encoding="UTF-8") as fIn:
    for line in tqdm(fIn, desc=f"Reading documents from {inp_path}"):
        try:
            doc = json.loads(line)
            doc_id, psg_id = doc["id"].split("-")
            vector = doc["vector"]
            doc_map[doc_id][psg_id] = vector
        except:
            print(line.split("\t")[0])
with open(out_path, "w", encoding="UTF-8") as fOut:
    for doc_id in tqdm(doc_map, desc=f"Saving re-index documents to: {out_path}"):
        for idx, psg_id in enumerate(sorted(doc_map[doc_id].keys())):
            new_doc_psg_id = doc_id + "-" + str(idx)
            doc_json = {"id": new_doc_psg_id, "vector": doc_map[doc_id][psg_id]}
            fOut.write(json.dumps(doc_json) + "\n")
