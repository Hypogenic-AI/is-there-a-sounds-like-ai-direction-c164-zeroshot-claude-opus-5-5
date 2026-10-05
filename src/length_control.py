"""Length-controlled detector check: re-score every main-grid text truncated to its first N words
(texts shorter than N words are dropped), so that shorter steered answers cannot look 'human' merely by length."""
import json
import numpy as np
import pandas as pd
from common import ROOT
from detectors import supervised_scores

N = 50
d = pd.read_json(f"{ROOT}/results/steering/main_scores.jsonl", lines=True)
d = d[d.n_words >= N].copy()
texts = [" ".join(t.split()[:N]) for t in d.text]
sc = supervised_scores(texts)
for k, v in sc.items():
    d[f"{k}_trunc{N}"] = v
d[["uid", "cond", "family", "alpha", f"desklib_trunc{N}", f"fakespot_trunc{N}", f"hc3rob_trunc{N}"]].to_csv(
    f"{ROOT}/results/steering_main_length_control.csv", index=False)
d["fam"] = d.family.str.replace(r"random\d", "random", regex=True)
print(d.groupby(["fam", "alpha"]).agg(n=("uid", "size"), desklib=(f"desklib_trunc{N}", "mean"), fakespot=(f"fakespot_trunc{N}", "mean")).round(3).to_string())
