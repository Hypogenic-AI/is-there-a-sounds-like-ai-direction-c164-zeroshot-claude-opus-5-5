"""Fit all directions (per layer) from cached activations and save them.

AI directions (diff-of-means, AI minus human, fit on TRAIN splits only):
  ai_hc3   : HC3 ChatGPT answers vs human answers to the same question
  ai_hap   : HAP instruct-model continuations (gpt-4o, gpt-4o-mini, llama-3 8B/70B-Instruct) vs human continuation of the same prefix
  ai       : unit(unit(ai_hc3) + unit(ai_hap))  <- main "AI-ness" direction
  ai_perp  : ai with the span of the six confound directions projected out
Confound directions (authorship held fixed within each contrast):
  formality : Llama-3.1-8B formal rewrite - informal rewrite of the same passage (both AI)
  register  : HAP human text, acad+news - spok+tvm+fic (both human)
  domain    : HC3 human answers, encyclopedic domains - reddit_eli5 (both human)
  fluency   : human texts, lowest-NLL tertile - highest-NLL tertile (both human)
  verbosity : HC3 human answers, longest tertile (full answer) - shortest tertile
  assistant : Assistant axis = default-Assistant responses - role-play responses (Lu et al. style)
usage: python directions.py {instruct|base}
"""
import sys, json
import numpy as np
from common import load_jsonl, ROOT

INST_GENS = {"gpt-4o", "gpt-4o-mini", "llama3-70b-inst", "llama3-8b-inst"}


def load(which, ds):
    z = np.load(f"{ROOT}/data/acts/{which}_{ds}.npz")
    rows = load_jsonl(f"{ROOT}/data/processed/{ds}.jsonl")
    return rows, z["acts"], z["nll"], z["ntok"]


def dmean(A, pos, neg):
    return A[pos].astype(np.float32).mean(0) - A[neg].astype(np.float32).mean(0)


def unitL(V):  # unit-normalise each layer's vector
    return V / (np.linalg.norm(V, axis=-1, keepdims=True) + 1e-12)


def main(which):
    D = {}
    rows, A, nll, _ = load(which, "hc3")
    ai = np.array([r["ai"] for r in rows]); tr = np.array([r["split"] == "train" for r in rows])
    dom = np.array([r["domain"] for r in rows])
    D["ai_hc3"] = dmean(A, tr & (ai == 1), tr & (ai == 0))
    hum = tr & (ai == 0)
    D["domain"] = dmean(A, hum & (dom != "reddit_eli5"), hum & (dom == "reddit_eli5"))
    wl = np.array([len(r["text"].split()) for r in rows])
    lo, hi = np.quantile(wl[hum], [1 / 3, 2 / 3])
    D["verbosity"] = dmean(A, hum & (wl >= hi), hum & (wl <= lo))
    nll_h = [(A[hum], nll[hum])]
    # projection stats on HC3 train for clamping
    rows2, A2, nll2, _ = load(which, "hap")
    gen = np.array([r["generator"] for r in rows2]); tr2 = np.array([r["split"] == "train" for r in rows2])
    dom2 = np.array([r["domain"] for r in rows2])
    isinst = np.isin(gen, list(INST_GENS))
    D["ai_hap"] = dmean(A2, tr2 & isinst, tr2 & (gen == "human"))
    h2 = tr2 & (gen == "human")
    D["register"] = dmean(A2, h2 & np.isin(dom2, ["acad", "news"]), h2 & np.isin(dom2, ["spok", "tvm", "fic"]))
    nll_h.append((A2[h2], nll2[h2]))
    Ah = np.concatenate([a for a, _ in nll_h]); nh = np.concatenate([n for _, n in nll_h])
    lo, hi = np.quantile(nh, [1 / 3, 2 / 3])
    D["fluency"] = dmean(Ah, nh <= lo, nh >= hi)
    rows3, A3, _, _ = load(which, "rewrites")
    st = np.array([r["style"] for r in rows3])
    D["formality"] = dmean(A3, st == "formal", st == "informal")
    try:
        D["assistant"] = np.load(f"{ROOT}/data/acts/{which}_assistant_axis_dir.npy")
    except FileNotFoundError:
        print("no assistant axis yet")
    U = {k: unitL(v) for k, v in D.items()}
    U["ai"] = unitL(U["ai_hc3"] + U["ai_hap"])
    # confound-orthogonal AI direction: u_ai with the span of all confound directions removed
    conf = [c for c in ["formality", "register", "domain", "fluency", "verbosity", "assistant"] if c in D]
    perp = np.zeros_like(U["ai"])
    for l in range(U["ai"].shape[0]):
        Q, _ = np.linalg.qr(np.stack([D[c][l] for c in conf]).T.astype(np.float64))
        perp[l] = U["ai"][l] - Q @ (Q.T @ U["ai"][l])
    U["ai_perp"] = unitL(perp)
    # mean projections of human / AI train texts on the main direction (per layer)
    proj_h = np.concatenate([np.einsum("nld,ld->nl", A[tr & (ai == 0)].astype(np.float32), U["ai"]),
                             np.einsum("nld,ld->nl", A2[h2].astype(np.float32), U["ai"])]).mean(0)
    proj_a = np.concatenate([np.einsum("nld,ld->nl", A[tr & (ai == 1)].astype(np.float32), U["ai"]),
                             np.einsum("nld,ld->nl", A2[tr2 & isinst].astype(np.float32), U["ai"])]).mean(0)
    ph2 = np.concatenate([np.einsum("nld,ld->nl", A[tr & (ai == 0)].astype(np.float32), U["ai_perp"]),
                          np.einsum("nld,ld->nl", A2[h2].astype(np.float32), U["ai_perp"])]).mean(0)
    pa2 = np.concatenate([np.einsum("nld,ld->nl", A[tr & (ai == 1)].astype(np.float32), U["ai_perp"]),
                          np.einsum("nld,ld->nl", A2[tr2 & isinst].astype(np.float32), U["ai_perp"])]).mean(0)
    raw_norm = {k: np.linalg.norm(v, axis=-1) for k, v in D.items()}
    np.savez(f"{ROOT}/data/acts/{which}_directions.npz", **{f"u_{k}": v for k, v in U.items()},
             **{f"raw_{k}": v for k, v in D.items()}, proj_h=proj_h, proj_a=proj_a, proj_h_perp=ph2, proj_a_perp=pa2)
    print(which, "saved", list(U))


if __name__ == "__main__":
    main(sys.argv[1])
