"""Build the text corpora used throughout the project.

Output: data/processed/{hc3,hap,mage,mage_gpt4}.jsonl with fields
  text, ai (1 = machine-written), dataset, group (pair id), domain, generator, split
plus data/processed/steer_prompts.jsonl (held-out HC3 questions for generation).
"""
import ast, json, os, random, re
import pandas as pd

random.seed(0)
OUT = "data/processed"
os.makedirs(OUT, exist_ok=True)


def detok(s: str) -> str:
    """Undo the PTB-style tokenisation in HC3 (applied to both sides, so it cannot
    act as an authorship cue)."""
    s = s.replace(" ", " ")
    s = re.sub(r" ([.,!?;:%)\]}])", r"\1", s)
    s = re.sub(r"([(\[{$]) ", r"\1", s)
    s = re.sub(r" (n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1", s)
    s = re.sub(r" ' (s|re|ve|ll|d|m|t)\b", r"'\1", s)
    s = re.sub(r'" ([^"]*?) "', r'"\1"', s)
    s = re.sub(r" - ", "-", s) if re.search(r"\w - \w", s) and s.count(" - ") > 3 else s
    s = re.sub(r"\s+", " ", s).strip()
    return s


def as_list(x):
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except Exception:
        return [x]


def write(name, rows):
    with open(f"{OUT}/{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(name, len(rows))


# ---------------- HC3: question-matched human vs ChatGPT answers -------------
caps = {"reddit_eli5": 900, "finance": 600, "medicine": 600, "open_qa": 600, "wiki_csai": 600}
hc3 = [json.loads(l) for l in open("data/HC3/all.jsonl")]
random.shuffle(hc3)
by_dom = {}
for ex in hc3:
    h, c = as_list(ex["human_answers"]), as_list(ex["chatgpt_answers"])
    if not h or not c:
        continue
    h, c = detok(h[0]), detok(c[0])
    if len(h.split()) < 30 or len(c.split()) < 30:
        continue
    by_dom.setdefault(ex["source"], []).append((detok(ex["question"]), h, c))
rows, prompts = [], []
for dom, items in by_dom.items():
    n_steer = 40 if dom in ("reddit_eli5", "open_qa", "finance", "medicine") else 0
    for i, (q, h, c) in enumerate(items[: caps[dom] + n_steer]):
        gid = f"hc3_{dom}_{i}"
        if i < n_steer:  # held out entirely: prompts for the steering experiments
            prompts.append(dict(id=gid, question=q, human=h, chatgpt=c, domain=dom))
            continue
        split = "train" if random.random() < 0.7 else "test"
        for t, ai in ((h, 0), (c, 1)):
            rows.append(dict(text=t, ai=ai, dataset="hc3", group=gid, domain=dom,
                             generator="chatgpt" if ai else "human", split=split, question=q))
write("hc3", rows)
write("steer_prompts", prompts)

# ---------------- HAP: prefix-matched human vs LLM continuations -------------
gens = {"gpt-4o-2024-08-06": "gpt-4o", "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
        "llama-3-70B-Instruct": "llama3-70b-inst", "llama-3-8B-Instruct": "llama3-8b-inst",
        "llama-3-70B": "llama3-70b-base", "llama-3-8B": "llama3-8b-base"}
P = "data/human-ai-parallel-corpus/text_data/hape-text_{}.parquet"
hum = pd.read_parquet(P.format("human-chunk-2"))
hum["doc"] = hum.doc_id.str.split("@").str[0]
docs = sorted(hum.doc.unique())
random.shuffle(docs)
docs = docs[:1800]
split_of = {d: ("train" if random.random() < 0.7 else "test") for d in docs}
hum = hum.set_index("doc")
rows = []
for d in docs:
    rows.append(dict(text=hum.loc[d, "text"], ai=0, dataset="hap", group=d, domain=d.split("_")[0],
                     generator="human", split=split_of[d]))
for f, g in gens.items():
    df = pd.read_parquet(P.format(f))
    df["doc"] = df.doc_id.str.split("@").str[0]
    df = df.set_index("doc")
    for d in docs:
        if d in df.index:
            rows.append(dict(text=df.loc[d, "text"], ai=1, dataset="hap", group=d, domain=d.split("_")[0],
                             generator=g, split=split_of[d]))
write("hap", rows)

# ---------------- MAGE: out-of-distribution, many generators -----------------
m = pd.read_csv("data/MAGE/test.csv").dropna()
m["ai"] = 1 - m.label
m["domain"] = m.src.str.split("_").str[0]
m["generator"] = m.src.apply(lambda s: "human" if s.endswith("human") else s.split("_machine_")[-1])
chat = m[m.generator.str.contains("gpt-3.5|davinci-003")]
old = m[(m.ai == 1) & ~m.generator.str.contains("gpt-3.5|davinci")]
hm = m[m.ai == 0]
sel = pd.concat([hm.sample(1200, random_state=0), chat.sample(600, random_state=0), old.sample(600, random_state=0)])
rows = [dict(text=r.text, ai=int(r.ai), dataset="mage", group=f"mage_{i}", domain=r.domain,
             generator=r.generator, split="test") for i, r in enumerate(sel.itertuples())]
write("mage", rows)
g4 = pd.read_csv("data/MAGE/test_ood_set_gpt.csv").dropna()
rows = [dict(text=r.text, ai=int(1 - r.label), dataset="mage_gpt4", group=f"g4_{i}", domain=r.src.split("_")[0],
             generator="gpt4" if r.label == 0 else "human", split="test") for i, r in enumerate(g4.itertuples())]
write("mage_gpt4", rows)
