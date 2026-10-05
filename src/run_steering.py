"""Generate steered outputs.
usage:
  python run_steering.py pilot                 # layer x strength pilot on 40 prompts (instruct)
  python run_steering.py main LAYER            # full condition grid on 160 held-out HC3 questions (instruct)
  python run_steering.py base LAYER            # base model: continuation of human text prefixes
Outputs results/steering/{name}.jsonl (id, cond, family, alpha, layer, question, text).
Strength a is in units of the human-AI gap along the AI direction at that layer (a=-1 moves
the residual stream by exactly the human-AI mean difference towards human); every control
vector gets the same absolute L2 norm.
"""
import json, os, sys, time
import numpy as np
import torch
from common import load_model, load_jsonl, ROOT
from steer import Steerer, generate, random_unit

OUT = f"{ROOT}/results/steering"
os.makedirs(OUT, exist_ok=True)
HUMAN_SYS = ("Answer the way an ordinary person would when replying on an online forum such as Reddit. "
             "Write casually and naturally, like a human, and do not sound like an AI assistant.")


def run(name, which, prompts, conds, chat=True, max_new_tokens=200):
    path = f"{OUT}/{name}.jsonl"
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["cond"] for l in open(path)}
    tok, model = load_model(which)
    D = np.load(f"{ROOT}/data/acts/{which}_directions.npz")
    st = Steerer(model)
    with open(path, "a") as f:
        for c in conds:
            if c["cond"] in done:
                continue
            t = time.time()
            st.clear()
            if c["kind"] == "add":
                l = c["layer"]
                gap = float(D["proj_a"][l] - D["proj_h"][l])
                v = c["vec"](D, l, model.config.hidden_size)
                st.add(l, c["alpha"] * gap * v)
            elif c["kind"] == "addall":  # shift every layer by alpha x its own human-AI gap
                for l in c["layers"]:
                    st.add(l, c["alpha"] * float(D["proj_a"][l] - D["proj_h"][l]) * D["u_ai"][l])
            elif c["kind"] == "clamp":
                for l in c["layers"]:
                    st.clamp(l, D["u_ai"][l], float(D[c["target"]][l]))
            outs = generate(tok, model, [p["question"] for p in prompts], system=c.get("system"),
                            max_new_tokens=max_new_tokens, chat=chat, seed=c.get("seed", 0))
            st.clear()
            for p, o in zip(prompts, outs):
                f.write(json.dumps(dict(id=p["id"], cond=c["cond"], family=c["family"], alpha=c.get("alpha", 0.0),
                                        layer=c.get("layer"), question=p["question"], text=o)) + "\n")
            f.flush()
            print(name, c["cond"], f"{time.time()-t:.0f}s", "|", outs[0][:150].replace("\n", " "), flush=True)
    del model; torch.cuda.empty_cache()


U = lambda key: (lambda D, l, d: D[key][l])
RAND = lambda seed: (lambda D, l, d: random_unit(d, 1000 * seed + l))


def add(family, vec, alpha, layer, **kw):
    return dict(cond=f"{family}@L{layer}a{alpha:+g}", family=family, kind="add", vec=vec, alpha=alpha, layer=layer, **kw)


if __name__ == "__main__":
    mode = sys.argv[1]
    P = load_jsonl(f"{ROOT}/data/processed/steer_prompts.jsonl")
    if mode == "pilot":
        prompts = [p for i, p in enumerate(P) if i % 4 == 0]  # 40 prompts, all domains
        conds = [dict(cond="none", family="none", kind="none")]
        for l in [8, 12, 16, 20, 24]:
            for a in [-8, -4, -2, 2, 4, 8]:
                conds.append(add("ai", U("u_ai"), a, l))
        run("pilot", "instruct", prompts, conds)
    elif mode == "main":
        L = int(sys.argv[2]); A = [float(x) for x in sys.argv[3].split(",")]
        conds = [dict(cond="none", family="none", kind="none"),
                 dict(cond="none_seed2", family="none_seed2", kind="none", seed=12345),
                 dict(cond="prompt_human", family="prompt_human", kind="none", system=HUMAN_SYS)]
        for a in A:
            conds.append(add("ai", U("u_ai"), a, L))
        for a in A:
            conds.append(add("ai", U("u_ai"), a, L, system=HUMAN_SYS) | dict(cond=f"ai+prompt@L{L}a{a:+g}", family="ai+prompt_human")) if a < 0 else None
        conds = [c for c in conds if c]
        for a in A:
            for s in range(3):
                conds.append(add(f"random{s}", RAND(s), a, L))
            for fam in ["ai_perp", "formality", "assistant", "fluency", "register"]:
                conds.append(add(fam, U(f"u_{fam}"), a, L))
        nl = 28
        conds.append(dict(cond="clamp_human", family="clamp_human", kind="clamp", layers=list(range(1, nl)), target="proj_h"))
        conds.append(dict(cond="clamp_ai", family="clamp_ai", kind="clamp", layers=list(range(1, nl)), target="proj_a"))
        for a in [-1.0, -0.5, -0.25, 0.25, 0.5, 1.0]:
            conds.append(dict(cond=f"addall_a{a:+g}", family="ai_all_layers", kind="addall", layers=list(range(1, nl)), alpha=a))
        run("main", "instruct", P, conds)
    elif mode == "base":
        L = int(sys.argv[2]); A = [float(x) for x in sys.argv[3].split(",")]
        hap = [r for r in load_jsonl(f"{ROOT}/data/processed/hap.jsonl") if r["generator"] == "human" and r["split"] == "test"]
        import pandas as pd
        c1 = pd.read_parquet(f"{ROOT}/data/human-ai-parallel-corpus/text_data/hape-text_human-chunk-1.parquet")
        c1["doc"] = c1.doc_id.str.split("@").str[0]; c1 = c1.set_index("doc")
        prompts = []
        for r in hap[:160]:
            words = c1.loc[r["group"], "text"].split()
            prompts.append(dict(id=r["group"], question=" ".join(words[-120:])))
        conds = [dict(cond="none", family="none", kind="none")]
        for a in A:
            conds.append(add("ai", U("u_ai"), a, L))
            conds.append(add("random0", RAND(0), a, L))
        conds.append(dict(cond="clamp_ai", family="clamp_ai", kind="clamp", layers=list(range(1, 28)), target="proj_a"))
        run("base", "base", prompts, conds, chat=False, max_new_tokens=160)
