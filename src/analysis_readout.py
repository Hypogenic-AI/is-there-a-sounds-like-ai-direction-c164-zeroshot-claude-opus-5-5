"""Read-out analysis: does the AI direction separate human vs AI text, does it transfer
across datasets / generators, is it distinct from the confound directions, and how does
it compare between base and instruct models?
usage: python analysis_readout.py   (needs directions for both models)
Writes results/readout_*.json/csv.
"""
import json, os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from directions import load, INST_GENS
from common import ROOT

R = f"{ROOT}/results"
os.makedirs(R, exist_ok=True)
MODELS = [m for m in ["instruct", "base"] if os.path.exists(f"{ROOT}/data/acts/{m}_directions.npz")]


def auc(s, y):
    return float(roc_auc_score(y, s))


def eval_sets(which):
    """name -> (acts [N,L,d] fp16, y, extra dict) for held-out evaluation."""
    S = {}
    rows, A, nll, ntok = load(which, "hc3")
    y = np.array([r["ai"] for r in rows]); te = np.array([r["split"] == "test" for r in rows])
    wl = np.array([len(r["text"].split()) for r in rows])
    S["HC3 (ChatGPT)"] = (A[te], y[te], dict(nll=nll[te], words=wl[te]))
    rows, A, nll, ntok = load(which, "hap")
    gen = np.array([r["generator"] for r in rows]); te = np.array([r["split"] == "test" for r in rows])
    wl = np.array([len(r["text"].split()) for r in rows])
    for g in ["gpt-4o", "gpt-4o-mini", "llama3-70b-inst", "llama3-8b-inst", "llama3-70b-base", "llama3-8b-base"]:
        m = te & np.isin(gen, ["human", g])
        S[f"HAP {g}"] = (A[m], (gen[m] != "human").astype(int), dict(nll=nll[m], words=wl[m]))
    rows, A, nll, ntok = load(which, "mage")
    gen = np.array([r["generator"] for r in rows]); y = np.array([r["ai"] for r in rows])
    chat = np.array(["gpt-3.5" in g or "davinci-003" in g for g in gen])
    wl = np.array([len(r["text"].split()) for r in rows])
    for name, m in [("MAGE chat (GPT-3.5/davinci-003)", (y == 0) | chat), ("MAGE older LMs", (y == 0) | ((y == 1) & ~chat))]:
        S[name] = (A[m], y[m], dict(nll=nll[m], words=wl[m]))
    rows, A, nll, ntok = load(which, "mage_gpt4")
    y = np.array([r["ai"] for r in rows]); wl = np.array([len(r["text"].split()) for r in rows])
    S["MAGE GPT-4 (OOD)"] = (A, y, dict(nll=nll, words=wl))
    rows, A, nll, ntok = load(which, "rewrites")
    st = np.array([r["style"] for r in rows]); m = np.isin(st, ["human", "neutral"])
    wl = np.array([len(r["text"].split()) for r in rows])
    S["Llama-3.1 paraphrase"] = (A[m], (st[m] == "neutral").astype(int), dict(nll=nll[m], words=wl[m]))
    return S


def train_xy(which, layer):
    X, Y = [], []
    rows, A, _, _ = load(which, "hc3")
    tr = np.array([r["split"] == "train" for r in rows])
    X.append(A[tr, layer]); Y.append(np.array([r["ai"] for r in rows])[tr])
    rows, A, _, _ = load(which, "hap")
    gen = np.array([r["generator"] for r in rows]); tr = np.array([r["split"] == "train" for r in rows])
    m = tr & (np.isin(gen, list(INST_GENS)) | (gen == "human"))
    X.append(A[m, layer]); Y.append((gen[m] != "human").astype(int))
    return np.concatenate(X).astype(np.float32), np.concatenate(Y)


def main():
    out = {}
    dirs = {m: np.load(f"{ROOT}/data/acts/{m}_directions.npz") for m in MODELS}
    sets = {m: eval_sets(m) for m in MODELS}
    L = dirs[MODELS[0]]["u_ai"].shape[0]
    # ---- 1. layer sweep of the diff-of-means directions -----------------------
    rec = []
    for m in MODELS:
        for name, (A, y, ex) in sets[m].items():
            for l in range(1, L):
                for dname in ["ai", "ai_hc3", "ai_hap", "ai_perp"]:
                    s = A[:, l].astype(np.float32) @ dirs[m][f"u_{dname}"][l]
                    rec.append(dict(model=m, eval=name, layer=l, direction=dname, auroc=auc(s, y)))
    sweep = pd.DataFrame(rec); sweep.to_csv(f"{R}/readout_layer_sweep.csv", index=False)
    # pick layer: best mean AUROC of the pooled direction across held-out sets (instruct)
    piv = sweep[(sweep.model == "instruct") & (sweep.direction == "ai")].groupby("layer").auroc.mean()
    best = int(piv.idxmax()); out["best_layer_instruct_readout"] = best
    print("mean AUROC by layer (instruct)", piv.round(3).to_dict())
    LAYERS = sorted(set([best, 8, 12, 16, 20]))
    out["layers_reported"] = LAYERS
    # ---- 2. probe vs diff-of-means and controls at reported layers -----------
    rec = []
    for m in MODELS:
        for l in LAYERS:
            Xtr, Ytr = train_xy(m, l)
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(C=0.01, max_iter=3000).fit(sc.transform(Xtr), Ytr)
            for name, (A, y, ex) in sets[m].items():
                X = A[:, l].astype(np.float32)
                r = dict(model=m, layer=l, eval=name, n=len(y),
                         dom_ai=auc(X @ dirs[m]["u_ai"][l], y),
                         probe=auc(clf.decision_function(sc.transform(X)), y),
                         neg_nll=auc(-ex["nll"], y), words=auc(ex["words"], y))
                for c in ["formality", "register", "domain", "fluency", "verbosity", "assistant"]:
                    if f"u_{c}" in dirs[m].files:
                        r[f"dir_{c}"] = auc(X @ dirs[m][f"u_{c}"][l], y)
                rec.append(r)
    tab = pd.DataFrame(rec); tab.to_csv(f"{R}/readout_auroc.csv", index=False)
    # ---- 3. geometry: cosines with confound directions ------------------------
    rec = []
    for m in MODELS:
        d = dirs[m]
        names = [k[2:] for k in d.files if k.startswith("u_")]
        for l in range(1, L):
            for a in names:
                for b in names:
                    if a < b:
                        rec.append(dict(model=m, layer=l, a=a, b=b, cos=float(d[f"u_{a}"][l] @ d[f"u_{b}"][l])))
    cos = pd.DataFrame(rec); cos.to_csv(f"{R}/readout_cosines.csv", index=False)
    # ---- 4. nullspace test: remove confound subspace, refit, re-evaluate -------
    rec = []
    rng = np.random.default_rng(0)
    for m in MODELS:
        d = dirs[m]
        conf = [c for c in ["formality", "register", "domain", "fluency", "verbosity", "assistant"] if f"u_{c}" in d.files]
        for l in LAYERS:
            Xtr, Ytr = train_xy(m, l)
            C = np.stack([d[f"raw_{c}"][l] for c in conf]).astype(np.float64)
            Q, _ = np.linalg.qr(C.T)  # d x k orthonormal basis of confound span
            Qr, _ = np.linalg.qr(rng.standard_normal(Q.shape))
            Qai = d["u_ai"][l][:, None]
            u0 = d["u_ai"][l]
            frac = float(np.sum((Q.T @ u0) ** 2))
            for tag, B in [("none", None), ("confounds", Q), ("random_k", Qr), ("ai_dir", Qai)]:
                P = (lambda X: X) if B is None else (lambda X, B=B: X - (X @ B) @ B.T)
                Xp = P(Xtr.astype(np.float64))
                u = Xp[Ytr == 1].mean(0) - Xp[Ytr == 0].mean(0)
                for name, (A, y, ex) in sets[m].items():
                    rec.append(dict(model=m, layer=l, removed=tag, k=0 if B is None else B.shape[1], eval=name,
                                    auroc=auc(P(A[:, l].astype(np.float64)) @ u, y), frac_ai_in_conf_span=frac))
    ns = pd.DataFrame(rec); ns.to_csv(f"{R}/readout_nullspace.csv", index=False)
    # ---- 5. scalar confounds: NLL / length partial out ------------------------
    rec = []
    for m, l in [(mm, ll) for mm in MODELS for ll in sorted({best, 12})]:
        for dn in ["ai", "ai_perp"]:
            for name, (A, y, ex) in sets[m].items():
                s = A[:, l].astype(np.float32) @ dirs[m][f"u_{dn}"][l]
                Z = np.stack([ex["nll"], np.log(ex["words"] + 1)], 1)
                # fit on half, evaluate on the other half to avoid in-sample optimism
                idx = np.random.default_rng(0).permutation(len(y)); a_, b_ = idx[: len(y) // 2], idx[len(y) // 2:]
                base = LogisticRegression(max_iter=2000).fit(Z[a_], y[a_])
                full = LogisticRegression(max_iter=2000).fit(np.c_[Z, s][a_], y[a_])
                r = dict(model=m, layer=l, direction=dn, eval=name,
                         auroc_nll_len=auc(base.decision_function(Z[b_]), y[b_]),
                         auroc_nll_len_plus_dir=auc(full.decision_function(np.c_[Z, s][b_]), y[b_]),
                         rho_dir_nll_human=float(spearmanr(s[y == 0], ex["nll"][y == 0])[0]),
                         rho_dir_nll_ai=float(spearmanr(s[y == 1], ex["nll"][y == 1])[0]),
                         rho_dir_len_human=float(spearmanr(s[y == 0], ex["words"][y == 0])[0]),
                         rho_dir_len_ai=float(spearmanr(s[y == 1], ex["words"][y == 1])[0]))
                beta = np.linalg.lstsq(np.c_[np.ones(len(s)), Z], s, rcond=None)[0]
                r["auroc_dir_resid"] = auc(s - np.c_[np.ones(len(s)), Z] @ beta, y)
                rec.append(r)
    pd.DataFrame(rec).to_csv(f"{R}/readout_scalar_confounds.csv", index=False)
    # ---- 6. base vs instruct ---------------------------------------------------
    if len(MODELS) == 2:
        rec = []
        di, db = dirs["instruct"], dirs["base"]
        for l in range(1, L):
            r = dict(layer=l, cos_ai_base_inst=float(di["u_ai"][l] @ db["u_ai"][l]),
                     cos_formality_base_inst=float(di["u_formality"][l] @ db["u_formality"][l]))
            if "u_assistant" in di.files and "u_assistant" in db.files:
                r["cos_assistant_base_inst"] = float(di["u_assistant"][l] @ db["u_assistant"][l])
            for name in sets["instruct"]:
                Ai, y, _ = sets["instruct"][name]; Ab, yb, _ = sets["base"][name]
                si = Ai[:, l].astype(np.float32) @ di["u_ai"][l]
                sb = Ab[:, l].astype(np.float32) @ db["u_ai"][l]
                sx = Ai[:, l].astype(np.float32) @ db["u_ai"][l]  # base direction read in instruct model
                dprime = lambda s, y: float((s[y == 1].mean() - s[y == 0].mean()) / np.sqrt(0.5 * (s[y == 1].var() + s[y == 0].var())))
                r[f"{name}|inst_auc"] = auc(si, y); r[f"{name}|base_auc"] = auc(sb, yb); r[f"{name}|base_dir_on_inst_auc"] = auc(sx, y)
                r[f"{name}|inst_d"] = dprime(si, y); r[f"{name}|base_d"] = dprime(sb, yb)
            rec.append(r)
        pd.DataFrame(rec).to_csv(f"{R}/readout_base_vs_instruct.csv", index=False)
    json.dump(out, open(f"{R}/readout_summary.json", "w"), indent=1)
    print(out)


if __name__ == "__main__":
    main()
