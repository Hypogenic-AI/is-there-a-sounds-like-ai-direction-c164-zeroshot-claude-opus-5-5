"""Batch chat completion with a local vLLM model (run inside .venv-vllm).
usage: python vllm_chat.py MODEL in.jsonl out.jsonl [temperature] [max_tokens]
in.jsonl rows: {"id": ..., "messages": [...]}; out rows: {"id": ..., "output": ...}"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf_cache"))
from vllm import LLM, SamplingParams

model, inp, out = sys.argv[1:4]
temp = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
mx = int(sys.argv[5]) if len(sys.argv) > 5 else 400
rows = [json.loads(l) for l in open(inp)]
llm = LLM(model=model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85, seed=0)
outs = llm.chat([r["messages"] for r in rows], SamplingParams(temperature=temp, top_p=0.95 if temp > 0 else 1.0,
                                                              max_tokens=mx, seed=0))
with open(out, "w") as f:
    for r, o in zip(rows, outs):
        f.write(json.dumps({"id": r["id"], "output": o.outputs[0].text}) + "\n")
print("wrote", len(rows))
