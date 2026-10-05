"""Activation steering during generation.

Interventions (applied at every position, prompt and generated tokens):
  add   : h_l <- h_l + a * gap_l * v       at one layer l, v a unit direction.
          gap_l = (mean AI projection - mean human projection) on the AI direction at layer l,
          i.e. a = -1 moves activations by one human-AI gap towards "human".
          Control directions (random, formality, assistant, ...) use the same absolute norm.
  clamp : h_l <- h_l + (c_l - h_l.u_l) u_l  at a set of layers, c_l = mean human (or AI) projection:
          "ablation" of the AI direction to the human level, with per-layer directions.
"""
import json, os, sys, argparse
import numpy as np
import torch
from common import load_model, load_jsonl, ROOT


class Steerer:
    def __init__(self, model):
        self.model = model
        self.handles = []

    def clear(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _hook(self, fn):
        def hook(mod, inp, out):
            if isinstance(out, tuple):
                return (fn(out[0]),) + tuple(out[1:])
            return fn(out)
        return hook

    def add(self, layer, vec):
        v = torch.tensor(vec, dtype=torch.bfloat16, device="cuda")
        # hidden_states[layer] is the output of decoder block layer-1
        self.handles.append(self.model.model.layers[layer - 1].register_forward_hook(self._hook(lambda h: h + v)))

    def clamp(self, layer, u, c):
        u = torch.tensor(u, dtype=torch.float32, device="cuda")
        def f(h):
            hf = h.float()
            return (hf + (c - hf @ u)[..., None] * u).to(h.dtype)
        self.handles.append(self.model.model.layers[layer - 1].register_forward_hook(self._hook(f)))


@torch.no_grad()
def generate(tok, model, prompts, system=None, max_new_tokens=200, bs=80, seed=0, chat=True):
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    outs = []
    for i in range(0, len(prompts), bs):
        batch = prompts[i:i + bs]
        if chat:
            texts = []
            for p in batch:
                m = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": p}]
                texts.append(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True))
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        else:
            enc = tok(batch, return_tensors="pt", padding=True).to("cuda")
        torch.manual_seed(seed + i)
        o = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9,
                           pad_token_id=tok.pad_token_id)
        outs += [tok.decode(x, skip_special_tokens=True).strip() for x in o[:, enc.input_ids.shape[1]:]]
    return outs


def random_unit(d, seed):
    r = np.random.default_rng(seed).standard_normal(d)
    return r / np.linalg.norm(r)
