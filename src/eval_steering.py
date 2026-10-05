"""Score steered generations with measures that are independent of the Qwen activations.
usage:
  python eval_steering.py detect NAME     # detectors + Binoculars + embeddings + stylometrics
  python eval_steering.py judgeprep NAME  # write judge requests (then run vllm_chat.py + judge.py parse)
Inputs results/steering/NAME.jsonl; outputs results/steering/NAME_scores.jsonl.
"""
import json, os, re, sys
import numpy as np
from common import load_jsonl, ROOT

S = f"{ROOT}/results/steering"
AI_PHRASES = ["it is important to", "it's important to", "overall,", "in summary", "in conclusion", "additionally",
              "however,", "generally", "it is worth noting", "it's worth noting", "there are several", "there are a few",
              "can help", "it's also", "it is also", "various", "ensure", "typically", "factors", "consult",
              "i hope this helps", "let me know", "here are", "key", "crucial", "essential", "delve"]


def style(t):
    w = t.split()
    n = max(len(w), 1)
    sents = [s for s in re.split(r"[.!?]+\s", t) if s.strip()]
    low = t.lower()
    return dict(n_words=len(w),
                sent_len=float(np.mean([len(s.split()) for s in sents])) if sents else 0.0,
                list_items=len(re.findall(r"(^|\n)\s*(\d+\.|[-*•])\s", t)),
                bold=t.count("**"),
                newlines=t.count("\n"),
                first_person=sum(x.lower().strip(",.!?") in ("i", "i'm", "i've", "me", "my", "i'd") for x in w) / n,
                contractions=len(re.findall(r"\b\w+'(t|s|re|ve|ll|d|m)\b", low)) / n,
                exclam=t.count("!") / n,
                ttr=len(set(x.lower() for x in w)) / n,
                ai_phrases=sum(low.count(p) for p in AI_PHRASES) / n * 100)


def detect(name):
    rows = load_jsonl(f"{S}/{name}.jsonl")
    if name == "main":  # reference texts: real human and ChatGPT answers to the same questions
        P = {p["id"]: p for p in load_jsonl(f"{ROOT}/data/processed/steer_prompts.jsonl")}
        for pid, p in P.items():
            rows.append(dict(id=pid, cond="ref_human", family="ref_human", alpha=0.0, layer=None, question=p["question"], text=p["human"]))
            rows.append(dict(id=pid, cond="ref_chatgpt", family="ref_chatgpt", alpha=0.0, layer=None, question=p["question"], text=p["chatgpt"]))
    texts = [r["text"] if r["text"].strip() else "." for r in rows]
    from detectors import supervised_scores, binoculars
    sc = supervised_scores(texts)
    sc.update(binoculars(texts))
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")
    E = emb.encode(texts, normalize_embeddings=True, batch_size=64)
    Q = emb.encode([r["question"] for r in rows], normalize_embeddings=True, batch_size=64)
    ref = {r["id"]: E[i] for i, r in enumerate(rows) if r["cond"] == "none"}
    with open(f"{S}/{name}_scores.jsonl", "w") as f:
        for i, r in enumerate(rows):
            d = dict(r, **{k: float(v[i]) for k, v in sc.items()}, **style(r["text"]))
            d["sim_unsteered"] = float(E[i] @ ref[r["id"]]) if r["id"] in ref else None
            d["sim_question"] = float(E[i] @ Q[i])
            d["uid"] = f"{r['cond']}|{r['id']}"
            f.write(json.dumps(d) + "\n")
    print("scored", len(rows))


def judgeprep(name):
    from judge import PROMPT
    rows = load_jsonl(f"{S}/{name}_scores.jsonl")
    with open(f"{S}/{name}_judge_req.jsonl", "w") as f:
        for r in rows:
            q = r["question"] if name != "base" else ("(Continue the following passage) ..." + " ".join(r["question"].split()[-60:]))
            t = " ".join(r["text"].split()[:300]) or "(empty)"
            f.write(json.dumps({"id": r["uid"], "messages": [{"role": "user", "content": PROMPT.format(q=q, t=t)}]}) + "\n")


if __name__ == "__main__":
    {"detect": detect, "judgeprep": judgeprep}[sys.argv[1]](sys.argv[2])
