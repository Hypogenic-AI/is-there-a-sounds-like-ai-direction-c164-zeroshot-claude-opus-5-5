import json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf_cache"))
MODELS = {"instruct": "Qwen/Qwen2.5-7B-Instruct", "base": "Qwen/Qwen2.5-7B"}
MAX_TOK = 128   # every text is truncated to at most this many tokens
SKIP = 1        # first position (attention sink, huge norm) excluded from pooling


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


def load_model(which="instruct", device="cuda"):
    name = MODELS[which]
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
    return tok, model


def token_budget(tok, rows):
    """Length control: all members of a group (same question / same prefix) are cut to
    the same number of tokens, the minimum over the group (capped at MAX_TOK)."""
    ids = [tok(r["text"], add_special_tokens=False)["input_ids"] for r in rows]
    by_group = {}
    for r, i in zip(rows, ids):
        by_group[r["group"]] = min(by_group.get(r["group"], MAX_TOK), len(i))
    return [i[: by_group[r["group"]]] for r, i in zip(rows, ids)]


@torch.no_grad()
def extract(tok, model, rows, bs=16, paired=True):
    """Mean-pooled residual stream at every layer + mean token NLL.
    Returns acts [N, L+1, d] float16, nll [N], ntok [N]."""
    ids = token_budget(tok, rows) if paired else [tok(r["text"], add_special_tokens=False)["input_ids"][:MAX_TOK] for r in rows]
    order = np.argsort([len(i) for i in ids])
    L = model.config.num_hidden_layers + 1
    d = model.config.hidden_size
    acts = np.zeros((len(rows), L, d), dtype=np.float16)
    nll = np.zeros(len(rows), dtype=np.float32)
    ntok = np.array([len(i) for i in ids])
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    for s in range(0, len(order), bs):
        idx = order[s: s + bs]
        mx = max(len(ids[i]) for i in idx)
        inp = torch.full((len(idx), mx), pad, dtype=torch.long)
        att = torch.zeros((len(idx), mx), dtype=torch.long)
        for j, i in enumerate(idx):
            inp[j, : len(ids[i])] = torch.tensor(ids[i])
            att[j, : len(ids[i])] = 1
        inp, att = inp.cuda(), att.cuda()
        out = model(input_ids=inp, attention_mask=att, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, 1).float()  # [b, L, T, d]
        m = att.clone().float()
        m[:, :SKIP] = 0
        pooled = (hs * m[:, None, :, None]).sum(2) / m.sum(1).clamp(min=1)[:, None, None]
        acts[idx] = pooled.cpu().numpy().astype(np.float16)
        lp = torch.log_softmax(out.logits[:, :-1].float(), -1)
        tl = -lp.gather(-1, inp[:, 1:, None])[..., 0]
        mm = att[:, 1:].float()
        nll[idx] = ((tl * mm).sum(1) / mm.sum(1).clamp(min=1)).cpu().numpy()
    return acts, nll, ntok


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-12)
