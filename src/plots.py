"""Figures for the paper (static PDF, matplotlib). usage: python plots.py [readout|steering|all]"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import ROOT

R = f"{ROOT}/results"; F = f"{ROOT}/paper_draft/figures"
os.makedirs(F, exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GRAY = "#8a8a85"
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "axes.edgecolor": "#9a9a95",
                     "lines.linewidth": 1.6, "legend.frameon": False, "figure.dpi": 200, "savefig.bbox": "tight"})
EVALS = ["HC3 (ChatGPT)", "HAP gpt-4o", "HAP llama3-8b-inst", "MAGE GPT-4 (OOD)", "MAGE chat (GPT-3.5/davinci-003)",
         "Llama-3.1 paraphrase", "HAP llama3-8b-base", "MAGE older LMs"]


def readout():
    s = pd.read_csv(f"{R}/readout_layer_sweep.csv")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for ax, dn, title in [(axs[0], "ai", "(a) AI direction $u_{AI}$ (diff-of-means)"),
                          (axs[1], "ai_perp", "(b) $u_{AI}^{\\perp}$ (confound span removed)")]:
        for i, e in enumerate(EVALS):
            g = s[(s.model == "instruct") & (s.direction == dn) & (s["eval"] == e)].sort_values("layer")
            ax.plot(g.layer, g.auroc, color=C[i], label=e, ls="--" if ("base" in e or "older" in e) else "-")
        ax.axhline(0.5, color=GRAY, lw=0.8); ax.set_title(title, fontsize=8.5); ax.set_xlabel("layer")
    axs[0].set_ylabel("held-out AUROC")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6.5, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.2))
    fig.savefig(f"{F}/readout_layers.pdf"); plt.close(fig)

    c = pd.read_csv(f"{R}/readout_cosines.csv")
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for i, k in enumerate(["formality", "register", "fluency", "verbosity", "domain", "assistant"]):
        g = c[(c.model == "instruct") & (((c.a == "ai") & (c.b == k)) | ((c.b == "ai") & (c.a == k)))].sort_values("layer")
        ax.plot(g.layer, g.cos, color=C[i], label=k)
    g = c[(c.model == "instruct") & (c.a == "ai_hap") & (c.b == "ai_hc3")].sort_values("layer")
    ax.plot(g.layer, g.cos, color="black", ls=":", label="HC3 vs HAP AI dirs")
    ax.axhline(0, color=GRAY, lw=0.8); ax.axvline(12, color=GRAY, lw=0.6, ls="--")
    ax.set_xlabel("layer"); ax.set_ylabel("cosine with $u_{AI}$"); ax.legend(fontsize=6.3, ncol=2, loc="lower left")
    fig.savefig(f"{F}/readout_cosines.pdf"); plt.close(fig)

    b = pd.read_csv(f"{R}/readout_base_vs_instruct.csv")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.4))
    for i, e in enumerate(["HC3 (ChatGPT)", "HAP gpt-4o", "MAGE GPT-4 (OOD)"]):
        axs[0].plot(b.layer, b[f"{e}|inst_auc"], color=C[i], label=f"{e} instruct")
        axs[0].plot(b.layer, b[f"{e}|base_auc"], color=C[i], ls="--", label=f"{e} base")
    axs[0].set_xlabel("layer"); axs[0].set_ylabel("AUROC of $u_{AI}$"); axs[0].legend(fontsize=6, ncol=1)
    axs[0].set_title("(a) read-out in base vs instruct", fontsize=8.5)
    axs[1].plot(b.layer, b.cos_ai_base_inst, color=C[0], label="$u_{AI}$")
    axs[1].plot(b.layer, b.cos_formality_base_inst, color=C[1], label="formality")
    axs[1].plot(b.layer, b.cos_assistant_base_inst, color=C[2], label="assistant axis")
    axs[1].set_ylim(0.8, 1.005); axs[1].set_xlabel("layer"); axs[1].set_ylabel("cos(base, instruct)")
    axs[1].set_title("(b) same direction in both models", fontsize=8.5); axs[1].legend(fontsize=6.5)
    fig.savefig(f"{F}/base_vs_instruct.pdf"); plt.close(fig)


FAMS = [("ai", "$u_{AI}$", C[0]), ("ai_perp", "$u_{AI}^{\\perp}$", C[6]), ("random", "random (3 seeds)", GRAY),
        ("formality", "formality", C[1]), ("fluency", "fluency", C[2]), ("register", "register", C[3]),
        ("assistant", "Assistant axis", C[4])]


def steering():
    t = pd.read_csv(f"{R}/steering_main_summary.csv")
    t["fam2"] = t.family.str.replace(r"random\d", "random", regex=True)
    ref = {k: t[t.cond == k].iloc[0] for k in ["none", "prompt_human", "ref_human", "ref_chatgpt"]}
    panels = [("desklib", "desklib P(AI)"), ("fakespot", "fakespot P(AI)"), ("bino_ai", "Binoculars AI-score (-B)"),
              ("coherence", "judge coherence (1-10)"), ("relevance", "judge relevance (1-10)"), ("falcon_nll", "Falcon-7B NLL")]
    fig, axs = plt.subplots(2, 3, figsize=(7.4, 4.4))
    for ax, (m, lab) in zip(axs.flat, panels):
        for fam, name, col in FAMS:
            g = t[t.fam2 == fam].groupby("alpha")[m].mean().reset_index()
            g = pd.concat([g, pd.DataFrame({"alpha": [0.0], m: [ref["none"][m]]})]).sort_values("alpha")
            ax.plot(g.alpha, g[m], color=col, marker="o", ms=2.5, label=name, lw=2.0 if fam == "ai" else 1.2,
                    ls="--" if fam == "random" else "-")
        for k, col, ls, nm in [("prompt_human", "black", ":", "prompt: 'write like a human'"),
                               ("ref_human", C[7], "-.", "real human answers"), ("ref_chatgpt", C[5], "-.", "real ChatGPT answers")]:
            ax.axhline(ref[k][m], color=col, ls=ls, lw=0.9, label=nm)
        ax.set_title(lab, fontsize=8); ax.set_xlabel("steering strength $\\alpha$ (human-AI gaps)")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(h, l, fontsize=6.5, loc="lower center", ncol=5)
    fig.savefig(f"{F}/steering_dose.pdf"); plt.close(fig)

    # trade-off: detector vs coherence, one point per condition (negative alphas + baselines)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, m, lab in [(axs[0], "desklib", "desklib P(AI)"), (axs[1], "bino_ai", "Binoculars AI-score (-B)")]:
        for fam, name, col in FAMS:
            g = t[(t.fam2 == fam) & (t.alpha < 0)].groupby("alpha")[[m, "coherence"]].mean()
            g = pd.concat([pd.DataFrame({m: [ref["none"][m]], "coherence": [ref["none"]["coherence"]]}), g.sort_index(ascending=False)])
            ax.plot(g.coherence, g[m], color=col, marker="o", ms=3, label=name, ls="--" if fam == "random" else "-")
        for k, mk, nm, col in [("prompt_human", "*", "prompt baseline", "black"), ("ref_human", "s", "real human", C[7]),
                               ("ref_chatgpt", "^", "real ChatGPT", C[5])]:
            ax.scatter(ref[k]["coherence"], ref[k][m], marker=mk, s=40, color=col, zorder=5, label=nm)
        for fam in ["clamp_human"]:
            r = t[t.cond == fam]
            if len(r):
                ax.scatter(r.coherence, r[m], marker="D", s=25, color=C[3], zorder=5, label="clamp to human mean (all layers)")
        ax.set_xlabel("judge coherence (1-10)"); ax.set_ylabel(lab); ax.invert_xaxis()
    h, l = axs[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.legend(h, l, fontsize=6.5, loc="lower center", ncol=6)
    fig.savefig(f"{F}/steering_tradeoff.pdf"); plt.close(fig)

    # stylometrics under u_AI steering
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 1.9))
    for ax, (m, lab) in zip(axs, [("n_words", "words"), ("list_items", "list items"), ("first_person", "1st-person rate"), ("ai_phrases", "AI-phrase rate (%)")]):
        for fam, name, col in FAMS[:3]:
            g = t[t.fam2 == fam].groupby("alpha")[m].mean().reset_index()
            g = pd.concat([g, pd.DataFrame({"alpha": [0.0], m: [ref["none"][m]]})]).sort_values("alpha")
            ax.plot(g.alpha, g[m], color=col, marker="o", ms=2.5, label=name, ls="--" if fam == "random" else "-")
        ax.axhline(ref["ref_human"][m], color=C[7], ls="-.", lw=0.9, label="real human")
        ax.set_title(lab, fontsize=8); ax.set_xlabel("$\\alpha$")
    axs[0].legend(fontsize=5.5)
    fig.tight_layout(); fig.savefig(f"{F}/steering_style.pdf"); plt.close(fig)


if __name__ == "__main__":
    w = sys.argv[1] if len(sys.argv) > 1 else "all"
    if w in ("readout", "all"):
        readout()
    if w in ("steering", "all"):
        steering()
