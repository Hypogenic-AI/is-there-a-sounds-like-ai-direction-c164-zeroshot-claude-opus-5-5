#!/bin/bash
# After main steering generation: base-model steering, scoring with detectors, LLM judge.
set -e
cd "$(dirname "$0")"
export CC=$PWD/../bin/cc HF_HOME=$PWD/../hf_cache VLLM_USE_FLASHINFER_SAMPLER=0
PY=../.venv/bin/python; PYV=../.venv-vllm/bin/python
$PY run_steering.py base 12 -8,-4,4,8
for n in main validation base; do $PY eval_steering.py detect $n; $PY eval_steering.py judgeprep $n; done
cat ../results/steering/{main,validation,base}_judge_req.jsonl > ../results/steering/all_judge_req.jsonl
$PYV vllm_chat.py google/gemma-3-12b-it ../results/steering/all_judge_req.jsonl ../results/steering/all_judge_out.jsonl 0.0 60
$PY judge.py parse ../results/steering/all_judge_out.jsonl ../results/steering/all_judge.jsonl
echo PIPELINE2 DONE
