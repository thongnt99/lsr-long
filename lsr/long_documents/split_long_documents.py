import sys
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer
from tqdm import tqdm

inp_path = sys.argv[1]
out_path = inp_path.replace("splits", "splits_psg")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
with open(inp_path, "r", encoding="UTF-8") as fIn, open(
    out_path, "w", encoding="UTF-8"
) as fOut:
    for line in tqdm(fIn):
        did, dtext = line.split("\t")
        num_tok_in_psg = 0
        psgs = [""]
        sentences = sent_tokenize(dtext)
        for sent in sentences:
            num_tok = len(tokenizer.tokenize(sent))
            if num_tok_in_psg + num_tok <= 400:
                psgs[-1] = psgs[-1] + " " + sent
                num_tok_in_psg += num_tok
            else:
                psgs.append(sent)
                num_tok_in_psg = num_tok
        psgs = [psg for psg in psgs if len(psg) > 0]
        for idx, psg in enumerate(psgs):
            psg_id = did + "@@" + str(idx)
            fOut.write(f"{psg_id}\t{psg}\n")

