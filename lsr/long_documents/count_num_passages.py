import json
import sys
from collections import defaultdict
from tqdm import tqdm

doc_map = defaultdict(dict)

inp_path = sys.argv[1]
with open(inp_path, "r", encoding="UTF-8") as fIn:
    for line in fIn:
        try:
            doc = json.loads(line)
            doc_id, psg_id = doc["id"].split("-")
            vector = doc["vector"]
            doc_map[doc_id][psg_id] = vector
        except:
            print(line.split("\t")[0])
max_psgs = max([len(doc_map[doc_id].keys()) for doc_id in doc_map])
print(max_psgs)
