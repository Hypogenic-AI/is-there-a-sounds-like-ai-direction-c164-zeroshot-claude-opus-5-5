"""Build LaTeX tables for the paper directly from results/*.csv (no hand-copied numbers)."""
import re
import numpy as np
import pandas as pd
from common import ROOT

R = f"{ROOT}/results"; T = f"{ROOT}/paper_draft/tables"
import os; os.makedirs(T, exist_ok=True)
f3 = lambda x: "--" if pd.isna(x) else f"{x:.2f}"
EV = [("HC3 (ChatGPT)", "HC3 ChatGPT$^\\dagger$"), ("HAP gpt-4o", "HAP GPT-4o$^\\dagger$"), ("HAP gpt-4o-mini", "HAP GPT-4o-mini$^\\dagger$"),
      ("HAP llama3-70b-inst", "HAP Llama-3-70B-Inst$^\\dagger$"), ("HAP llama3-8b-inst", "HAP Llama-3-8B-Inst$^\\dagger$"),
      ("MAGE GPT-4 (OOD)", "MAGE GPT-4"), ("MAGE chat (GPT-3.5/davinci-003)", "MAGE GPT-3.5 / davinci-003"),
      ("Llama-3.1 paraphrase", "Llama-3.1 paraphrase (content-matched)"),
      ("HAP llama3-70b-base", "HAP Llama-3-70B \\emph{base}"), ("HAP llama3-8b-base", "HAP Llama-3-8B \\emph{base}"),
      ("MAGE older LMs", "MAGE older LMs (OPT, T5, ...)")]


def readout():
    a = pd.read_csv(f"{R}/readout_auroc.csv"); a = a[a.model == "instruct"]
    s = pd.read_csv(f"{R}/readout_layer_sweep.csv"); s = s[(s.model == "instruct")]
    n = pd.read_csv(f"{R}/readout_nullspace.csv"); n = n[(n.model == "instruct") & (n.layer == 12)]
    rows = []
    for key, name in EV:
        r12 = a[(a.layer == 12) & (a["eval"] == key)].iloc[0]
        r27 = a[(a.layer == 27) & (a["eval"] == key)].iloc[0]
        perp = s[(s.layer == 12) & (s.direction == "ai_perp") & (s["eval"] == key)].auroc.iloc[0]
        conf = max(r12[c] for c in ["dir_formality", "dir_register", "dir_fluency", "dir_assistant", "dir_domain"])
        nn = n[n["eval"] == key].set_index("removed").auroc
        rows.append(f"{name} & {f3(r12.dom_ai)} & {f3(perp)} & {f3(r27.dom_ai)} & {f3(r12.probe)} & {f3(r12.neg_nll)} & {f3(conf)} & {f3(nn['confounds'])} & {f3(nn['random_k'])} & {f3(nn['ai_dir'])}\\\\")
        if key == "HAP llama3-8b-inst" or key == "Llama-3.1 paraphrase":
            rows.append("\\midrule")
    open(f"{T}/readout.tex", "w").write("\n".join(rows) + "\n")


def validation():
    v = pd.read_csv(f"{R}/measure_validation_auroc.csv")
    names = {"val_hc3_ai": "HC3: ChatGPT vs human", "val_hap_gpt-4o": "HAP: GPT-4o vs human",
             "val_hap_llama3-8b-inst": "HAP: Llama-3-8B-Inst vs human", "val_hap_llama3-8b-base": "HAP: Llama-3-8B base vs human"}
    rows = [f"{names[r.set]} & {f3(r.desklib)} & {f3(r.fakespot)} & {f3(r.bino_ai)} & {f3(r.hc3rob)} & {f3(r.ai_likeness)}\\\\" for r in v.itertuples()]
    open(f"{T}/validation.tex", "w").write("\n".join(rows) + "\n")


def steering():
    t = pd.read_csv(f"{R}/steering_main_summary.csv").set_index("cond")
    lc = pd.read_csv(f"{R}/steering_main_length_control.csv")
    sc = pd.read_json(f"{R}/steering/main_scores.jsonl", lines=True)[["uid", "text"]]
    sc["en"] = sc.text.apply(lambda x: len(re.findall(r"[぀-ヿ㐀-鿿가-힯]", x)) / max(len(x), 1) <= 0.05)
    lc = lc.merge(sc[["uid", "en"]], on="uid"); lc = lc[lc.en]
    lcm = lc.groupby("cond")[["desklib_trunc50", "fakespot_trunc50"]].mean()
    conds = [("none", "unsteered"), ("none_seed2", "unsteered, other seed"), ("prompt_human", "prompt: ``write like a human''"),
             ("ai@L12a-2", "$u_{AI}$, $\\alpha{=}{-}2$"), ("ai@L12a-4", "$u_{AI}$, $\\alpha{=}{-}4$"), ("ai@L12a-6", "$u_{AI}$, $\\alpha{=}{-}6$"), ("ai@L12a-8", "$u_{AI}$, $\\alpha{=}{-}8$"),
             ("ai_perp@L12a-4", "$u_{AI}^{\\perp}$, $\\alpha{=}{-}4$"), ("ai_perp@L12a-6", "$u_{AI}^{\\perp}$, $\\alpha{=}{-}6$"),
             ("ai+prompt@L12a-4", "prompt + $u_{AI}$, $\\alpha{=}{-}4$"),
             ("addall_a-0.25", "$u_{AI}$ all layers, $\\alpha{=}{-}0.25$"),
             ("RANDOM-6", "random (3 seeds), $\\alpha{=}{-}6$"), ("RANDOM-8", "random (3 seeds), $\\alpha{=}{-}8$"),
             ("formality@L12a-8", "formality, $\\alpha{=}{-}8$"), ("fluency@L12a-8", "fluency, $\\alpha{=}{-}8$"),
             ("register@L12a-8", "register, $\\alpha{=}{-}8$"), ("assistant@L12a-6", "Assistant axis, $\\alpha{=}{-}6$"),
             ("clamp_human", "clamp to human mean, all layers"),
             ("ai@L12a+4", "$u_{AI}$, $\\alpha{=}{+}4$"), ("ai@L12a+8", "$u_{AI}$, $\\alpha{=}{+}8$"),
             ("ref_human", "\\emph{real human answers}"), ("ref_chatgpt", "\\emph{real ChatGPT answers}")]
    cols = ["frac_non_english", "desklib", "fakespot", "bino_ai", "desklib_logit", "coherence", "relevance", "sim_unsteered", "n_words"]
    rows = []
    for c, name in conds:
        if c.startswith("RANDOM"):
            a = c[6:]
            sub = t[t.index.str.match(rf"random\d@L12a{a}$")]
            r = sub[cols].mean(); l = lcm.loc[sub.index].mean()
        else:
            r = t.loc[c, cols]; l = lcm.loc[c] if c in lcm.index else pd.Series({"desklib_trunc50": np.nan, "fakespot_trunc50": np.nan})
        if c in ("ai@L12a-2", "ai_perp@L12a-4", "ai+prompt@L12a-4", "RANDOM-6", "clamp_human", "ai@L12a+4", "ref_human"):
            rows.append("\\midrule")
        rows.append(f"{name} & {r.frac_non_english*100:.0f} & {f3(r.desklib)} & {f3(r.fakespot)} & {f3(l.desklib_trunc50)} & {f3(l.fakespot_trunc50)} & {f3(r.bino_ai)} & {r.desklib_logit:.1f} & {r.coherence:.1f} & {r.relevance:.1f} & {f3(r.sim_unsteered)} & {r.n_words:.0f}\\\\")
    open(f"{T}/steering.tex", "w").write("\n".join(rows) + "\n")

    # matched-quality analysis: individual generations with judge coherence>=8 and relevance>=8
    d = pd.read_json(f"{R}/steering/main_scores.jsonl", lines=True)
    j = pd.read_json(f"{R}/steering/all_judge.jsonl", lines=True).drop_duplicates("id").set_index("id")
    d["coherence"] = d.uid.map(j.coherence); d["relevance"] = d.uid.map(j.relevance)
    d = d.merge(sc[["uid", "en"]], on="uid"); d = d[d.en]
    d["fam"] = d.family.str.replace(r"random\d", "random", regex=True)
    q = d[(d.coherence >= 8) & (d.relevance >= 8)]
    rows = []
    for fam, name in [("ai", "$u_{AI}$"), ("ai_perp", "$u_{AI}^{\\perp}$"), ("random", "random"), ("formality", "formality"), ("fluency", "fluency"), ("register", "register"), ("assistant", "Assistant axis")]:
        cells = []
        for a in [-4, -6, -8]:
            g = q[(q.fam == fam) & (q.alpha == a)]
            tot = len(d[(d.fam == fam) & (d.alpha == a)])
            cells.append(f"{g.desklib.mean():.2f} & {g.fakespot.mean():.2f} & {100*len(g)/max(tot,1):.0f}\\%" if len(g) else "-- & -- & 0\\%")
        rows.append(f"{name} & " + " & ".join(cells) + "\\\\")
    for c, name in [("none", "unsteered"), ("prompt_human", "prompt baseline"), ("ref_human", "real human answers")]:
        g = q[q.cond == c]; tot = len(d[d.cond == c])
        rows.append(f"{name} & \\multicolumn{{9}}{{c}}{{desklib {g.desklib.mean():.2f}, fakespot {g.fakespot.mean():.2f}, kept {100*len(g)/tot:.0f}\\%}}\\\\")
    open(f"{T}/matched_quality.tex", "w").write("\n".join(rows) + "\n")


def base():
    t = pd.read_csv(f"{R}/steering_base_summary.csv").set_index("cond")
    rows = []
    for c, name in [("none", "unsteered"), ("ai@L12a-4", "$u_{AI}^{base}$, $\\alpha{=}{-}4$"), ("ai@L12a-8", "$u_{AI}^{base}$, $\\alpha{=}{-}8$"),
                    ("random0@L12a-4", "random, $\\alpha{=}{-}4$"), ("random0@L12a-8", "random, $\\alpha{=}{-}8$"),
                    ("ai@L12a+4", "$u_{AI}^{base}$, $\\alpha{=}{+}4$"), ("ai@L12a+8", "$u_{AI}^{base}$, $\\alpha{=}{+}8$"),
                    ("random0@L12a+4", "random, $\\alpha{=}{+}4$"), ("random0@L12a+8", "random, $\\alpha{=}{+}8$")]:
        r = t.loc[c]
        rows.append(f"{name} & {f3(r.desklib)} & {f3(r.fakespot)} & {f3(r.bino_ai)} & {r.desklib_logit:.1f} & {r.ai_phrases:.2f} & {r.coherence:.1f} & {r.n_words:.0f}\\\\")
    open(f"{T}/base_steering.tex", "w").write("\n".join(rows) + "\n")


if __name__ == "__main__":
    readout(); validation(); steering(); base()
    print("tables written")


def pilot():
    d = pd.read_json(f"{R}/steering/pilot_scores.jsonl", lines=True)
    d["en"] = d.text.apply(lambda x: len(re.findall(r"[぀-ヿ㐀-鿿가-힯]", x)) / max(len(x), 1) <= 0.05)
    d["layer"] = d.layer.fillna(0).astype(int)
    rows = []
    for (l, a), g in d.groupby(["layer", "alpha"]):
        if l == 0:
            continue
        rows.append(f"{l} & {a:+g} & {100*(1-g.en.mean()):.0f} & {g.desklib.mean():.2f} & {g.fakespot.mean():.2f} & {g.bino_ai.mean():.2f} & {g.falcon_nll.mean():.2f} & {g.sim_unsteered.mean():.2f} & {g.n_words.mean():.0f}\\\\")
    g = d[d.cond == "none"]
    rows.insert(0, f"-- & 0 & 0 & {g.desklib.mean():.2f} & {g.fakespot.mean():.2f} & {g.bino_ai.mean():.2f} & {g.falcon_nll.mean():.2f} & 1.00 & {g.n_words.mean():.0f}\\\\\\midrule")
    open(f"{T}/pilot.tex", "w").write("\n".join(rows) + "\n")


pilot()


def steer_full():
    a = pd.read_csv(f"{R}/steering_main_all_summary.csv").set_index("cond")
    e = pd.read_csv(f"{R}/steering_main_summary.csv").set_index("cond")
    rows = []
    order = ["none", "prompt_human"] + [f"ai@L12a{x:+g}" for x in [-8, -6, -4, -2, 2, 4, 6, 8]] + \
            [f"ai_perp@L12a{x:+g}" for x in [-8, -6, -4, 4, 8]] + [f"ai+prompt@L12a{x:+g}" for x in [-8, -6, -4, -2]] + \
            [f"{f}@L12a{x:+g}" for f in ["formality", "fluency", "register", "assistant", "random0", "random1", "random2"] for x in [-8, -4, 4, 8]] + \
            [f"addall_a{x:+g}" for x in [-1, -0.5, -0.25, 0.25, 0.5, 1]] + ["clamp_human", "clamp_ai", "ref_human", "ref_chatgpt"]
    for c in order:
        r, s = a.loc[c], e.loc[c]
        name = c.replace("_", "\\_").replace("@L12a", " $\\alpha$=")
        rows.append(f"{name} & {r.frac_non_english*100:.0f} & {r.desklib:.2f} & {r.fakespot:.2f} & {r.bino_ai:.2f} & {s.desklib:.2f} & {s.fakespot:.2f} & {s.ai_likeness:.1f} & {s.coherence:.1f} & {s.relevance:.1f} & {s.falcon_nll:.2f}\\\\")
    open(f"{T}/steer_full.tex", "w").write("\n".join(rows) + "\n")


def scalar():
    s = pd.read_csv(f"{R}/readout_scalar_confounds.csv")
    s = s[(s.model == "instruct") & (s.layer == 12)]
    rows = []
    for key, name in EV:
        a = s[(s["eval"] == key) & (s.direction == "ai")].iloc[0]; b = s[(s["eval"] == key) & (s.direction == "ai_perp")].iloc[0]
        rows.append(f"{name} & {f3(a.auroc_nll_len)} & {f3(a.auroc_nll_len_plus_dir)} & {f3(a.auroc_dir_resid)} & {f3(b.auroc_nll_len)} & {f3(b.auroc_nll_len_plus_dir)} & {f3(b.auroc_dir_resid)}\\\\")
    open(f"{T}/scalar.tex", "w").write("\n".join(rows) + "\n")


steer_full()
if "direction" in pd.read_csv(f"{R}/readout_scalar_confounds.csv").columns:
    scalar()
