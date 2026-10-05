"""Content-matched control corpora made by rewriting human text with a local LLM
(Llama-3.1-8B-Instruct via vLLM -- a different family from the Qwen model we probe).

For each human source passage:
  formal   - very formal register
  informal - very casual register
  neutral  - a plain paraphrase (content fixed, author becomes an LLM)
formal-vs-informal gives a FORMALITY direction with authorship held fixed (both AI);
human-vs-neutral gives an AUTHORSHIP contrast with content held fixed.
usage: python make_rewrites.py prep   (writes the request file)
       python make_rewrites.py collect (after vllm_chat.py has run)
"""
import json, random, sys
from common import load_jsonl, ROOT

random.seed(1)
REQ = f"{ROOT}/data/processed/rewrite_requests.jsonl"
OUT = f"{ROOT}/data/processed/rewrite_outputs.jsonl"
INSTR = {
    "formal": "Rewrite the following passage in a highly formal, academic register. Keep the same content and roughly the same length.",
    "informal": "Rewrite the following passage in a very casual, informal, conversational register (as if texting a friend). Keep the same content and roughly the same length.",
    "neutral": "Paraphrase the following passage. Keep the same content, meaning and roughly the same length.",
}


def sources():
    hap = [r for r in load_jsonl(f"{ROOT}/data/processed/hap.jsonl") if r["ai"] == 0 and r["split"] == "train"]
    hc3 = [r for r in load_jsonl(f"{ROOT}/data/processed/hc3.jsonl") if r["ai"] == 0 and r["split"] == "train"]
    return random.sample(hap, 300) + random.sample(hc3, 200)


if sys.argv[1] == "prep":
    with open(REQ, "w") as f:
        for i, r in enumerate(sources()):
            for style, ins in INSTR.items():
                msg = f"{ins} Output only the rewritten passage, with no preamble or comments.\n\nPassage:\n{' '.join(r['text'].split()[:200])}"
                f.write(json.dumps({"id": f"{i}|{style}", "messages": [{"role": "user", "content": msg}]}) + "\n")
else:
    src = sources()
    outs = {r["id"]: r["output"] for r in load_jsonl(OUT)}
    rows = []
    for i, r in enumerate(src):
        rows.append(dict(text=" ".join(r["text"].split()[:200]), style="human", group=f"rw_{i}",
                         source_dataset=r["dataset"], domain=r["domain"], ai=0))
        for style in INSTR:
            o = outs[f"{i}|{style}"].strip()
            if o.lower().startswith(("here is", "here's")) and "\n" in o:
                o = o.split("\n", 1)[1].strip()
            rows.append(dict(text=o, style=style, group=f"rw_{i}", source_dataset=r["dataset"], domain=r["domain"], ai=1))
    with open(f"{ROOT}/data/processed/rewrites.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(len(rows))
