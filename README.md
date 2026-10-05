# Is there a "sounds like AI" direction in the residual stream?

Paper: [`paper_draft/main.pdf`](paper_draft/main.pdf) (LaTeX source in `paper_draft/`).

## Question
Human vs. AI text is known to be linearly separable in LLM activations. We ask:
1. **Causal?** Does adding or subtracting the human-vs-AI direction during generation make the model's *own* output read as more or less AI-written to *independent* detectors, while keeping coherence and content?
2. **Distinct?** Is it an "AI-ness" direction, or a relabelling of formality, fluency (perplexity), register, domain, verbosity or the Assistant persona?
3. **Base vs. instruct?** Does it exist in the pretrained base model, and does chat tuning amplify it?

## What we did
* **Model:** Qwen2.5-7B-Instruct and Qwen2.5-7B (base). Residual stream mean-pooled over ≤128 tokens of raw text. Every matched group (same question / same prefix) is cut to the same token length.
* **Data:** HC3 (question-matched human vs ChatGPT; PTB artefacts detokenised on both sides), the human-AI parallel corpus (prefix-matched human vs GPT-4o / GPT-4o-mini / Llama-3-Instruct / Llama-3 base), MAGE (OOD, 30+ generators) and MAGE-GPT4. Also 500 human passages rewritten by Llama-3.1-8B-Instruct in formal, informal and paraphrase versions.
* **Directions:** diff-of-means `u_AI` (HC3 + HAP pooled), plus six authorship-controlled confound directions: formality, register, domain, fluency, verbosity, and the Assistant axis (default Assistant vs. 24 role-play personas). `u_AI⊥` is `u_AI` with the confound span projected out.
* **Steering:** activation addition at layer 12 with strength in units of the human–AI gap, on 160 held-out HC3 questions. Equal-norm controls: 3 random directions, the confound directions, `u_AI⊥`, a "write like a human" system prompt, prompt+steer, multi-layer addition and clamping. Also steering of the base model on 160 human text prefixes.
* **Independent measures** (none sees Qwen activations): desklib and fakespot supervised detectors, Binoculars (Falcon-7B/-Instruct), a Gemma-3-12B judge (coherence/relevance), Falcon NLL, mpnet similarity to the unsteered answer, and stylometrics. All are validated on real held-out human/AI text.

## Main findings
* **Read-out:** `u_AI` gives AUROC 0.83–0.86 (mid layers) and up to 0.98 (L27) on held-out matched sets. It transfers to MAGE GPT-4 and to content-matched paraphrases. It does **not** detect base-model or pre-chat LM text (0.49–0.63), so it encodes **chat-assistant style**, not machine authorship.
* **Disentanglement:** about 54% of `u_AI`'s squared norm (L12) lies in the confound span (cos ≈0.68 formality, 0.69 fluency, 0.40 Assistant axis). Removing that span makes the read-out **better** (0.97–0.99), and `u_AI⊥` alone reads out at 0.97–0.99.
* **Causal:** subtracting `u_AI` at L12 lowers P(AI) from 0.99 to 0.42/0.34 (desklib/fakespot, α=−6). Binoculars moves to the human level, and the effect survives length control. Equal-norm random, formality, fluency and register directions stay at 0.95–1.00; the Assistant axis reaches 0.72 (α=−6) at a large coherence cost. The "write like a human" prompt leaves 99% flagged. Among answers the judge rates ≥8 on both coherence and relevance, `u_AI⊥` at α=−6 gives 0.40/0.30, matching real human answers (0.36/0.26). Adding the direction produces stock assistant prose.
* **Base vs. instruct:** base and instruct directions have cosine ≥0.98 at every layer with equal read-out AUROC, so chat tuning did not create or amplify the direction. Steering the base model with its own direction works in both directions.
* **Caveats:** the strongest settings cost coherence and sometimes switch the output to Chinese (excluded from the primary analysis). Clamping or ablating at all layers breaks the model. The LLM judge **cannot** tell human from AI text (AUROC ≤0.60), so AI-ness claims rest on the detectors and Binoculars.

## Layout
```
src/
  prepare_data.py      corpora -> data/processed/*.jsonl
  make_rewrites.py     formal/informal/paraphrase rewrites (Llama-3.1-8B via vLLM)
  extract_acts.py      mean-pooled residual activations, all layers (instruct|base)
  assistant_axis.py    persona generations + Assistant axis
  directions.py        all directions per layer (incl. u_AI_perp, projection stats)
  analysis_readout.py  layer sweep, probes, cosines, nullspace test, NLL/length controls, base vs instruct
  steer.py, run_steering.py   steering hooks; pilot / main grid / base-model experiment
  detectors.py, eval_steering.py, judge.py, vllm_chat.py   independent measures
  make_validation.py   real-text validation set for the measures
  length_control.py    detectors on first-50-word truncations
  analysis_steering.py summaries with bootstrap CIs, paired Wilcoxon tests, matched coherence
  make_tables.py, plots.py    LaTeX tables and figures for the paper
  pipeline2.sh         base steering -> detection -> judge
results/               CSV summaries; results/steering/*.jsonl = every generation with all scores
paper_draft/           main.tex, sections/, tables/ (generated), figures/ (generated), references.bib
logs/                  run logs
```
`data/`, `hf_cache/`, `.venv*/` and `bin/` hold large or local files and are git-ignored.

## Reproduce
```bash
uv venv .venv -p 3.12 && uv pip install --python .venv/bin/python torch==2.8.0 transformers==4.56.2 "sentence-transformers<5.2" accelerate datasets scikit-learn pandas matplotlib scipy pymupdf
uv venv .venv-vllm -p 3.12 && uv pip install --python .venv-vllm/bin/python vllm ziglang   # judge / rewrites
# no system C compiler here: bin/cc wraps `python -m ziglang cc` for Triton (export CC=$PWD/bin/cc), and VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HOME=$PWD/hf_cache   # needs HF_TOKEN with access to gemma-3 / Llama-3.1
cd src
python prepare_data.py
python make_rewrites.py prep && ../.venv-vllm/bin/python vllm_chat.py meta-llama/Llama-3.1-8B-Instruct ../data/processed/rewrite_requests.jsonl ../data/processed/rewrite_outputs.jsonl 0.7 400 && python make_rewrites.py collect
for m in instruct base; do python extract_acts.py $m; python assistant_axis.py $m; python directions.py $m; done
python analysis_readout.py
python run_steering.py pilot && python eval_steering.py detect pilot
python run_steering.py main 12 -8,-6,-4,-2,2,4,6,8
python make_validation.py && ./pipeline2.sh
python length_control.py && python analysis_steering.py && python make_tables.py && python plots.py all
cd ../paper_draft && latexmk -pdf main.tex
```
Hardware used: 1× RTX A6000 (48 GB). The full pipeline takes about 4 GPU-hours.
