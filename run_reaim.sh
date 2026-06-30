#!/bin/bash
set -e
export HF_HOME=/antidistillation-sampling/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=".venv/bin/accelerate launch"
TEACHER="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
PROXY="Qwen/Qwen2.5-3B"
STUDENT="meta-llama/Llama-3.2-3B"
EXP="math_repl"
SEED=42; LAM=0.0868; EPS=0.01; TAU=0.6

# Back up the frozen gradient before overwriting
[ -e ${EXP}/student_grads_frozen.pt ] || cp ${EXP}/student_grads.pt ${EXP}/student_grads_frozen.pt

# REAIM 1: distill the PROXY on the frozen poison traces
if [ ! -e ${EXP}/models/proxy_reaim/final ]; then
  echo "=== REAIM 1: distill proxy on poison traces ==="
  $PY distill.py student=${PROXY} tokenizer=${PROXY}-Instruct exp_dir=${EXP} \
    train_traces=${EXP}/traces/train_poison holdout_traces=${EXP}/traces/holdout \
    max_length=2048 batch_size=16 per_device_batch_size=2 num_epochs=3 do_eval=false \
    model_name=proxy_reaim wandb=false
fi

# REAIM 2: recompute gradient against updated proxy (overwrites student_grads.pt)
echo "=== REAIM 2: recompute reaimed gradient ==="
.venv/bin/python save_grad.py ${EXP}/traces/holdout.yaml --proxy_student=${EXP}/models/proxy_reaim/final

# REAIM 3: poison traces with reaimed gradient
if [ ! -e ${EXP}/traces/train_reaim ]; then
  echo "=== REAIM 3: poison traces (reaimed gradient) ==="
  $PY gentraces.py teacher=${TEACHER} proxy_student=${PROXY} exp_dir=${EXP} seed=${SEED} \
    data_split=hendrycks_math_train grad_path=${EXP}/student_grads.pt max_samples=1000 batch_size=16 \
    max_length=2048 max_prompt_length=1024 tau=${TAU} lam=${LAM} eps=${EPS} \
    trace_name=train_reaim use_wandb=false
fi

# REAIM 4: distill student on reaimed poison
if [ ! -e ${EXP}/models/student_reaim/final ]; then
  echo "=== REAIM 4: distill student (reaimed poison) ==="
  $PY distill.py student=${STUDENT} tokenizer=${STUDENT}-Instruct exp_dir=${EXP} \
    train_traces=${EXP}/traces/train_reaim holdout_traces=${EXP}/traces/holdout \
    max_length=2048 batch_size=16 per_device_batch_size=2 num_epochs=3 do_eval=false \
    model_name=student_reaim wandb=false
fi

# REAIM 5: eval reaimed-poisoned student
echo "=== REAIM 5: eval reaimed student ==="
$PY gentraces.py teacher=${EXP}/models/student_reaim/final is_teacher=false exp_dir=${EXP} seed=${SEED} \
  data_split=hendrycks_math_test max_samples=500 batch_size=64 \
  max_length=2048 max_prompt_length=1024 trace_name=eval_reaim use_wandb=false

echo "=== H1 RESULTS: frozen vs reaimed ==="
echo "CLEAN baseline:"
grep -A2 "stats:" ${EXP}/traces/eval_train_clean.yaml | grep accuracy
echo "FROZEN poison:"
grep -A2 "stats:" ${EXP}/traces/eval_train_poison.yaml | grep accuracy
echo "REAIMED poison:"
grep -A2 "stats:" ${EXP}/traces/eval_reaim.yaml | grep accuracy
