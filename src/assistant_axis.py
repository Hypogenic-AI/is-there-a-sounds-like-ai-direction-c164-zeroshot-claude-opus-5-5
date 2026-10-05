"""Assistant-axis control (after Lu et al., 2026, simplified).

The instruct model answers a set of questions (a) as its default Assistant and
(b) while role-playing 24 non-assistant characters. We record the residual stream
mean-pooled over the RESPONSE tokens (inside the chat template) at every layer.
Assistant axis = mean(default) - mean(role-play), per layer.
We also save the generated texts so they can be read as raw text later.
usage: python assistant_axis.py {instruct|base}
For the base model we reuse the instruct model's texts (teacher-forced), so the
base-model axis is measured on identical tokens.
"""
import json, os, sys
import numpy as np
import torch
from common import load_model, ROOT, load_jsonl

which = sys.argv[1] if len(sys.argv) > 1 else "instruct"
PERSONAS = ["a pirate captain from the 1700s", "a medieval peasant farmer", "a grumpy old fisherman",
            "a teenage skateboarder", "a Shakespearean actor who always speaks in verse", "a hard-boiled 1940s detective",
            "a cowboy on the American frontier", "a wise forest spirit", "a Viking warrior",
            "a sarcastic stand-up comedian", "a nervous first-year medical student", "a Victorian aristocrat",
            "a surfer dude from California", "an ancient Roman senator", "a conspiracy theorist who distrusts everything",
            "a kindergarten teacher talking to children", "a gruff army drill sergeant", "a dreamy romantic poet",
            "a street vendor in Mumbai", "a tired night-shift nurse", "a hip-hop artist",
            "a fantasy wizard", "a dog who can talk", "a ghost haunting an old house"]
QUESTIONS = ["What do you think about the weather today?", "How should I spend my weekend?",
             "What is the best way to cook an egg?", "Why do people like music?",
             "Can you tell me about yourself?", "What advice would you give a young person?",
             "How do you deal with stress?", "What makes a good friend?",
             "Why is the sky blue?", "What is your favorite food and why?",
             "How do I fix a flat bicycle tyre?", "Is money important for happiness?",
             "What do you do when you can't sleep?", "Explain how a bank loan works.",
             "What's the meaning of life?", "How should I prepare for a job interview?"]
TXT = f"{ROOT}/data/processed/assistant_axis_texts.jsonl"
tok, model = load_model("instruct" if which == "instruct" else "base")
L = model.config.num_hidden_layers + 1

if which == "instruct" and not os.path.exists(TXT):
    itok = tok
    recs = []
    conds = [("default", None)] + [(p, p) for p in PERSONAS]
    torch.manual_seed(0)
    for cname, p in conds:
        msgs_list = []
        for q in QUESTIONS:
            m = [] if p is None else [{"role": "system", "content": f"You are {p}. Stay fully in character at all times; never mention being an AI or an assistant."}]
            m.append({"role": "user", "content": q})
            msgs_list.append(m)
        prompts = [itok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs_list]
        itok.padding_side = "left"
        enc = itok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=160, do_sample=True, temperature=0.7, top_p=0.9,
                                 pad_token_id=itok.eos_token_id)
        for q, m, o in zip(QUESTIONS, msgs_list, out[:, enc.input_ids.shape[1]:]):
            recs.append(dict(cond=cname, question=q, messages=m, response=itok.decode(o, skip_special_tokens=True).strip()))
    with open(TXT, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

recs = load_jsonl(TXT)
itok = load_model.__globals__["AutoTokenizer"].from_pretrained("Qwen/Qwen2.5-7B-Instruct")
acts = np.zeros((len(recs), L, model.config.hidden_size), dtype=np.float32)
for i, r in enumerate(recs):
    prompt = itok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True)
    p_ids = itok(prompt, add_special_tokens=False)["input_ids"]
    r_ids = itok(r["response"], add_special_tokens=False)["input_ids"][:160]
    ids = torch.tensor([p_ids + r_ids]).cuda()
    with torch.no_grad():
        hs = model(input_ids=ids, output_hidden_states=True).hidden_states
    acts[i] = torch.stack([h[0, len(p_ids):].float().mean(0) for h in hs]).cpu().numpy()
cond = np.array([r["cond"] for r in recs])
np.savez(f"{ROOT}/data/acts/{which}_assistant_axis.npz", acts=acts.astype(np.float16), cond=cond)
axis = acts[cond == "default"].mean(0) - acts[cond != "default"].mean(0)
np.save(f"{ROOT}/data/acts/{which}_assistant_axis_dir.npy", axis)
print("saved", axis.shape)
