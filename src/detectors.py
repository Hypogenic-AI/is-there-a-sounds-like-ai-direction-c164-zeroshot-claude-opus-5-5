"""Independent AI-text detectors (none of them sees Qwen activations).

supervised: three public fine-tuned classifiers, P(AI)
  desklib  : desklib/ai-text-detector-v1.01 (DeBERTa-v3-large, trained on recent LLM text)
  fakespot : fakespot-ai/roberta-base-ai-text-detection-v1
  hc3rob   : Hello-SimpleAI/chatgpt-detector-roberta (trained on HC3; reported, not used as primary)
zero-shot: Binoculars (Hans et al., 2024) with Falcon-7B (observer) / Falcon-7B-Instruct (performer).
  bino_ai = -Binoculars score (higher = more AI-like). Also returns falcon_nll (mean token NLL of
  the text under Falcon-7B), used as a model-independent fluency/coherence proxy.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel, PreTrainedModel, AutoModelForSequenceClassification, AutoModelForCausalLM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf_cache"))


class DesklibAIDetectionModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        h = self.model(input_ids, attention_mask=attention_mask)[0]
        m = attention_mask.unsqueeze(-1).float()
        return self.classifier((h * m).sum(1) / m.sum(1).clamp(min=1e-9))


@torch.no_grad()
def supervised_scores(texts, bs=32):
    out = {}
    tok = AutoTokenizer.from_pretrained("desklib/ai-text-detector-v1.01")
    m = DesklibAIDetectionModel.from_pretrained("desklib/ai-text-detector-v1.01").cuda().eval()
    s = []
    for i in range(0, len(texts), bs):
        e = tok(texts[i:i + bs], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
        s += torch.sigmoid(m(e.input_ids, e.attention_mask).float()).squeeze(-1).tolist()
    out["desklib"] = np.array(s); del m
    for key, name in [("fakespot", "fakespot-ai/roberta-base-ai-text-detection-v1"),
                      ("hc3rob", "Hello-SimpleAI/chatgpt-detector-roberta")]:
        tok = AutoTokenizer.from_pretrained(name)
        m = AutoModelForSequenceClassification.from_pretrained(name).cuda().eval().half()
        s = []
        for i in range(0, len(texts), bs):
            e = tok(texts[i:i + bs], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
            s += torch.softmax(m(**e).logits.float(), -1)[:, 1].tolist()
        out[key] = np.array(s); del m
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def binoculars(texts, bs=8, max_len=256):
    tok = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")
    tok.pad_token = tok.eos_token
    obs = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b", torch_dtype=torch.bfloat16).cuda().eval()
    perf = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct", torch_dtype=torch.bfloat16).cuda().eval()
    B, NLL = [], []
    for i in range(0, len(texts), bs):
        e = tok([t if t.strip() else "." for t in texts[i:i + bs]], padding=True, truncation=True,
                max_length=max_len, return_tensors="pt").to("cuda")
        lo = obs(**e).logits[:, :-1].float()
        lp = perf(**e).logits[:, :-1].float()
        tgt = e.input_ids[:, 1:]
        m = e.attention_mask[:, 1:].float()
        nll = torch.nn.functional.cross_entropy(lp.transpose(1, 2), tgt, reduction="none")  # performer ppl
        nll_obs = torch.nn.functional.cross_entropy(lo.transpose(1, 2), tgt, reduction="none")
        xent = -(torch.softmax(lo, -1) * torch.log_softmax(lp, -1)).sum(-1)  # cross-perplexity
        ppl = (nll * m).sum(1) / m.sum(1).clamp(min=1)
        x = (xent * m).sum(1) / m.sum(1).clamp(min=1)
        B += (ppl / x).tolist()
        NLL += ((nll_obs * m).sum(1) / m.sum(1).clamp(min=1)).tolist()
    del obs, perf
    torch.cuda.empty_cache()
    return {"bino_ai": -np.array(B), "falcon_nll": np.array(NLL)}
