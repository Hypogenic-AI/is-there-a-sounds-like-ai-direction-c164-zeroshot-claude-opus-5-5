"""Extract mean-pooled residual-stream activations (all layers) for every corpus.
usage: python extract_acts.py {instruct|base} ds1 ds2 ...
Saves data/acts/{model}_{ds}.npz with acts [N, L+1, d] (fp16), nll, ntok."""
import os, sys, time
import numpy as np
from common import load_model, load_jsonl, extract, ROOT

which = sys.argv[1]
dss = sys.argv[2:] or ["hc3", "hap", "mage", "mage_gpt4", "rewrites"]
os.makedirs(f"{ROOT}/data/acts", exist_ok=True)
tok, model = load_model(which)
for ds in dss:
    out = f"{ROOT}/data/acts/{which}_{ds}.npz"
    if os.path.exists(out):
        print("skip", out); continue
    rows = load_jsonl(f"{ROOT}/data/processed/{ds}.jsonl")
    t = time.time()
    # mage sets are unpaired: plain truncation to MAX_TOK
    acts, nll, ntok = extract(tok, model, rows, paired=not ds.startswith("mage"))
    np.savez(out, acts=acts, nll=nll, ntok=ntok)
    print(which, ds, acts.shape, f"{time.time()-t:.0f}s", flush=True)
