"""LLM-judge (Gemma-3-12B-it, local via vLLM; a different model family from Qwen,
the detectors and the rewriter).
usage: python judge.py prep  in.jsonl req.jsonl     (rows need id, question, text)
       python judge.py parse req_out.jsonl parsed.jsonl
"""
import json, re, sys

PROMPT = """You are an expert at distinguishing human-written text from text written by AI chatbots (like ChatGPT, Claude, Llama or Qwen).

Below is a question and a response to it. Rate the RESPONSE on three 1-10 scales:
- ai_likeness: how likely the response was written by an AI chatbot rather than a human (for example a person answering on a forum or Q&A site). 1 = certainly human, 10 = certainly AI.
- coherence: is it fluent, grammatical, well-formed and logically coherent text? 1 = gibberish or broken/repetitive, 10 = perfectly coherent.
- relevance: does it actually address the question? 1 = unrelated, 10 = directly and fully on topic.

Question: {q}

Response:
\"\"\"
{t}
\"\"\"

Answer with only a JSON object of the form {{"ai_likeness": <int>, "coherence": <int>, "relevance": <int>}}."""


def prep(inp, out):
    with open(out, "w") as f:
        for l in open(inp):
            r = json.loads(l)
            t = " ".join(r["text"].split()[:300]) or "(empty)"
            f.write(json.dumps({"id": r["id"], "messages": [{"role": "user", "content": PROMPT.format(q=r["question"], t=t)}]}) + "\n")


def parse(inp, out):
    bad = 0
    with open(out, "w") as f:
        for l in open(inp):
            r = json.loads(l)
            d = {}
            for k in ["ai_likeness", "coherence", "relevance"]:
                pat = "ai_like(?:li)?ness" if k == "ai_likeness" else k  # judge sometimes writes 'ai_likeliness'
                m = re.search(rf'"{pat}"\s*:\s*(\d+)', r["output"])
                d[k] = int(m.group(1)) if m else None
            bad += any(v is None for v in d.values())
            f.write(json.dumps({"id": r["id"], **d}) + "\n")
    print("unparsed", bad)


if __name__ == "__main__":
    {"prep": prep, "parse": parse}[sys.argv[1]](*sys.argv[2:])
