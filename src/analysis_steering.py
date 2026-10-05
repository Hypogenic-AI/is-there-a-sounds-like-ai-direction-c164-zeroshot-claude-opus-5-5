"""Aggregate steering results: per-condition means with bootstrap CIs, paired tests against
equal-norm random directions, matched-coherence comparison, validation of the measures.
usage: python analysis_steering.py
"""
import json, os, re, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from common import ROOT

S = f"{ROOT}/results/steering"
R = f"{ROOT}/results"
MEAS = ["non_english", "desklib", "fakespot", "desklib_logit", "fakespot_logit", "hc3rob", "bino_ai", "falcon_nll", "ai_likeness", "coherence", "relevance",
        "sim_unsteered", "sim_question", "n_words", "sent_len", "list_items", "bold", "newlines", "first_person",
        "contractions", "exclam", "ttr", "ai_phrases"]


def load(name):
    d = pd.read_json(f"{S}/{name}_scores.jsonl", lines=True)
    j = pd.read_json(f"{S}/all_judge.jsonl", lines=True).drop_duplicates("id").set_index("id")
    for k in ["ai_likeness", "coherence", "relevance"]:
        d[k] = d.uid.map(j[k]) if k in j else np.nan
    d["desklib_flag"] = (d.desklib > 0.5).astype(float)
    # strong negative steering sometimes switches the answer to Chinese; English-trained detectors are not
    # valid there, so the primary analysis uses English outputs only (CJK characters <= 5% of the text)
    d["non_english"] = d.text.apply(lambda t: len(re.findall(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", t)) / max(len(t), 1) > 0.05).astype(float)
    eps = 1e-6  # detector logits: still informative where probabilities saturate near 1
    d["desklib_logit"] = np.log((d.desklib + eps) / (1 - d.desklib + eps))
    d["fakespot_logit"] = np.log((d.fakespot + eps) / (1 - d.fakespot + eps))
    d["fakespot_flag"] = (d.fakespot > 0.5).astype(float)
    return d


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarize(d, name):
    nonen = d.groupby("cond").non_english.mean()
    if not name.endswith("_all"):
        d = d[d.non_english == 0]
    rec = []
    meas = MEAS + ["desklib_flag", "fakespot_flag"]
    for (cond, fam, a), g in d.groupby(["cond", "family", "alpha"], dropna=False):
        r = dict(cond=cond, family=fam, alpha=a, n=len(g), frac_non_english=float(nonen[cond]))
        for m in meas:
            if m in g and g[m].notna().any():
                r[m] = float(g[m].mean())
                lo, hi = boot_ci(g[m])
                r[m + "_lo"], r[m + "_hi"] = lo, hi
        rec.append(r)
    t = pd.DataFrame(rec)
    t.to_csv(f"{R}/steering_{name}_summary.csv", index=False)
    return t


def paired_tests(d, name, layer=12):
    """AI direction vs equal-norm random directions (mean over 3 seeds), and vs each control, per alpha."""
    rec = []
    piv = {m: d.pivot_table(index="id", columns="cond", values=m) for m in ["desklib", "fakespot", "desklib_logit", "bino_ai", "ai_likeness", "coherence", "relevance", "falcon_nll", "ai_phrases"]}
    alphas = sorted(d[d.family == "ai"].alpha.unique())
    for a in alphas:
        tag = f"@L{layer}a{a:+g}"
        for m, P in piv.items():
            ai = P.get("ai" + tag)
            rnd = P[[c for c in P.columns if c.startswith("random") and c.endswith(tag)]].mean(1)
            base = P["none"]
            for other_name, other in [("random(mean of 3)", rnd), ("none", base)] + \
                    [(f, P.get(f + tag)) for f in ["ai_perp", "formality", "assistant", "fluency", "register"]]:
                if ai is None or other is None:
                    continue
                ok = ai.notna() & other.notna()
                diff = (ai[ok] - other[ok])
                p = float(wilcoxon(ai[ok], other[ok]).pvalue) if (diff != 0).any() else 1.0
                rec.append(dict(alpha=a, measure=m, vs=other_name, mean_ai=float(ai[ok].mean()),
                                mean_other=float(other[ok].mean()), diff=float(diff.mean()), p_wilcoxon=p, n=int(ok.sum())))
    t = pd.DataFrame(rec); t.to_csv(f"{R}/steering_{name}_paired_tests.csv", index=False)
    return t


def matched_coherence(t, name, tol=1.0):
    """For each steering family: the strongest strength whose mean judge coherence AND relevance stay within
    `tol` points of the unsteered baseline; report detector scores there."""
    base = t[t.cond == "none"].iloc[0]
    rec = []
    for fam, g in t.groupby("family"):
        for sign in [-1, 1]:
            gg = g[(np.sign(g.alpha) == sign)]
            ok = gg[(gg.coherence >= base.coherence - tol) & (gg.relevance >= base.relevance - tol)]
            if len(ok) == 0 and fam not in ("prompt_human", "clamp_human", "clamp_ai", "none", "ref_human", "ref_chatgpt"):
                continue
            if fam in ("prompt_human", "clamp_human", "clamp_ai", "ref_human", "ref_chatgpt", "none", "none_seed2"):
                if sign == 1:
                    continue
                ok = g
            row = ok.loc[ok.alpha.abs().idxmax()] if len(ok) else None
            if row is None:
                continue
            rec.append(dict(family=fam, alpha=row.alpha, coherence=row.coherence, relevance=row.relevance,
                            desklib=row.desklib, desklib_flag=row.desklib_flag, fakespot=row.fakespot, bino_ai=row.bino_ai,
                            ai_likeness=row.ai_likeness, falcon_nll=row.falcon_nll, sim_unsteered=row.sim_unsteered))
    m = pd.DataFrame(rec); m.to_csv(f"{R}/steering_{name}_matched_coherence.csv", index=False)
    return m


def validation():
    d = load("validation")
    rec = []
    for fam, g in d.groupby("family"):
        for gen in sorted(g.cond.unique()):
            if gen.endswith("human"):
                continue
            hum = g[g.cond.str.endswith("human")]
            ai = g[g.cond == gen]
            x = pd.concat([hum, ai]); y = (x.cond == gen).astype(int)
            r = dict(set=gen)
            for m in ["desklib", "fakespot", "hc3rob", "bino_ai", "ai_likeness", "falcon_nll"]:
                ok = x[m].notna()
                r[m] = float(roc_auc_score(y[ok], x[m][ok])) if ok.sum() > 10 else np.nan
            rec.append(r)
    v = pd.DataFrame(rec); v.to_csv(f"{R}/measure_validation_auroc.csv", index=False)
    # agreement between measures on validation + main
    return v


if __name__ == "__main__":
    out = {}
    v = validation(); print(v.round(3).to_string())
    d = load("main")
    summarize(d, "main_all")
    t = summarize(d, "main")
    d = d[d.non_english == 0]
    cols = ["cond", "desklib", "desklib_flag", "fakespot", "bino_ai", "ai_likeness", "coherence", "relevance", "falcon_nll", "sim_unsteered", "n_words", "list_items", "first_person", "contractions"]
    print(t[cols].round(3).to_string())
    pt = paired_tests(d, "main")
    mc = matched_coherence(t, "main"); print(mc.round(3).to_string())
    # correlations between measures across all main generations
    corr = d[["desklib", "fakespot", "hc3rob", "bino_ai", "ai_likeness", "coherence", "falcon_nll"]].corr(method="spearman")
    corr.to_csv(f"{R}/steering_main_measure_corr.csv")
    if os.path.exists(f"{S}/base_scores.jsonl"):
        b = load("base")
        summarize(b, "base_all")
        tb = summarize(b, "base")
        b = b[b.non_english == 0]; print(tb[cols].round(3).to_string())
        paired_tests(b, "base")
