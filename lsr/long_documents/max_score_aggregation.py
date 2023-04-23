import sys
from tqdm import tqdm
from collections import defaultdict

inp_path = sys.argv[1]
out_path = sys.argv[2]
num_psg = int(sys.argv[3])
if len(sys.argv) > 4:
    sep = sys.argv[4]
else:
    sep = "@@"
in_list = defaultdict(set)
with open(inp_path, "r", encoding="UTF-8") as fIn, open(
    out_path, "w", encoding="UTF-8"
) as fOut:
    for line in tqdm(fIn):
        cols = line.strip().split(" ")
        id_text = cols[2]
        doc_id, psg_id = id_text.split(sep)
        qid = cols[0]
        if (
            doc_id in in_list[qid]
            or len(in_list[qid]) >= 1000
            or int(psg_id) >= num_psg
        ):
            continue
        else:
            in_list[qid].add(doc_id)
            cols[2] = doc_id
            fOut.write(" ".join(cols) + "\n")
