import json
import sys
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import scipy

doc_map = defaultdict(dict)

inp_path = sys.argv[1]
aggregation = sys.argv[2]
num_psg = int(sys.argv[3])
if len(sys.argv) > 4:
    sep = sys.argv[4]
else:
    sep = "-"
out_path = inp_path.replace("/doc", "/doc_" + aggregation)
with open(inp_path, "r", encoding="UTF-8") as fIn:
    for line in tqdm(fIn, desc=f"Reading collection from: {inp_path}"):
        try:
            doc = json.loads(line)
            doc_id, psg_id = doc["id"].split(sep)
            vector = doc["vector"]
            doc_map[doc_id][psg_id] = vector
        except:
            print(line.split("\t")[0])
lengths = []
max_passages = []
with open(out_path, "w", encoding="UTF-8") as fOut:
    for doc_id in tqdm(
        doc_map, desc=f"Aggregating and writing collection to: {out_path}"
    ):
        vector = defaultdict(lambda: 0)
        psg_ids = sorted(doc_map[doc_id])
        if aggregation == "first":
            first_p = psg_ids[0]
            vector = doc_map[doc_id][first_p]
        elif aggregation == "max":
            for psg_id in psg_ids[:num_psg]:
                for term in doc_map[doc_id][psg_id]:
                    vector[term] = max(vector[term], doc_map[doc_id][psg_id][term])
        elif aggregation == "sum":
            for psg_id in psg_ids[:num_psg]:
                for term in doc_map[doc_id][psg_id]:
                    vector[term] = vector[term] + doc_map[doc_id][psg_id][term]
        elif aggregation == "mean":
            real_num_psgs = len(psg_ids[:num_psg])
            for psg_id in psg_ids[:num_psg]:
                for term in doc_map[doc_id][psg_id]:
                    vector[term] = vector[term] + doc_map[doc_id][psg_id][term]
            for term in vector:
                vector[term] = vector[term] / real_num_psgs
        elif aggregation == "exp":
            real_num_psgs = len(psg_ids[:num_psg])
            for psg_id in psg_ids[:num_psg]:
                for term in doc_map[doc_id][psg_id]:
                    vector[term] = vector[term] + np.expm1(
                        doc_map[doc_id][psg_id][term]
                    )
            for term in vector:
                vector[term] = np.ceil(np.log1p(vector[term] / real_num_psgs))
        elif aggregation == "gmean":
            vector = defaultdict(list)
            for psg_id in psg_ids[:num_psg]:
                for term in doc_map[doc_id][psg_id]:
                    vector[term].append(doc_map[doc_id][psg_id][term])
            for term in vector:
                vector[term] = np.ceil(scipy.stats.gmean(vector[term]))
        else:
            raise Exception(f"Agg: {aggregation} not defined")

        doc = {"id": doc_id, "vector": vector}
        lengths.append(len(vector))
        fOut.write(json.dumps(doc))
print(f"Finished. Average length: {sum(lengths)/len(lengths)}")
