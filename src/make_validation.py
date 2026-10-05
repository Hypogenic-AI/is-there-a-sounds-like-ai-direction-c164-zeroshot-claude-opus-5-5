"""Validation set for the independent measures: real human vs real AI text (held-out test splits).
Writes results/steering/validation.jsonl in the same format as the steering outputs."""
import json, random
from common import load_jsonl, ROOT
random.seed(3)
rows = []
hc3 = [r for r in load_jsonl(f"{ROOT}/data/processed/hc3.jsonl") if r["split"] == "test"]
groups = sorted({r["group"] for r in hc3}); random.shuffle(groups); keep = set(groups[:250])
for r in hc3:
    if r["group"] in keep:
        rows.append(dict(id=r["group"], cond="val_hc3_" + ("ai" if r["ai"] else "human"), family="val_hc3", alpha=float(r["ai"]),
                         layer=None, question=r["question"], text=" ".join(r["text"].split()[:200])))
hap = [r for r in load_jsonl(f"{ROOT}/data/processed/hap.jsonl") if r["split"] == "test"]
docs = sorted({r["group"] for r in hap}); random.shuffle(docs); keep = set(docs[:150])
for r in hap:
    if r["group"] in keep and r["generator"] in ("human", "gpt-4o", "llama3-8b-inst", "llama3-8b-base"):
        rows.append(dict(id=r["group"], cond="val_hap_" + r["generator"], family="val_hap", alpha=float(r["ai"]),
                         layer=None, question="(Continue the passage)", text=" ".join(r["text"].split()[:200])))
with open(f"{ROOT}/results/steering/validation.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print(len(rows))
