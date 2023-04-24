import ir_measures
from ir_measures import *
from scipy import stats
import sys
import json

run1 = sys.argv[1]
run2 = sys.argv[2]
qrel = sys.argv[3]

qrels = list(ir_measures.read_trec_qrels(qrel))
# run1 = ir_measures.read_trec_run(run1)
run1 = json.load(open(run1))
run2 = ir_measures.read_trec_run(run2)
scores1 = {}
for metric in ir_measures.iter_calc([MRR @ 10], qrels, run1):
    scores1[metric.query_id] = metric.value

scores2 = {}
for metric in ir_measures.iter_calc([MRR @ 10], qrels, run2):
    scores2[metric.query_id] = metric.value
l1 = []
l2 = []
for qid in scores1:
    l1.append(scores1[qid])
    l2.append(scores2[qid])
result = stats.ttest_rel(l1, l2)
print(result)
