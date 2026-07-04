#!/bin/bash
set -e
export HF_HOME=/root/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=".venv/bin/accelerate launch"
EXP="gsm_repl"                      # SEPARATE exp dir on persistent volume
TEACHER="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
PROXY="Qwen/Qwen2.5-3B"
SEED=42; EPS=0.0001; TAU=0.6

clear_gpu () { pkill -9 -f gentraces 2>/dev/null || true; pkill -9 -f distill 2>/dev/null || true; pkill -9 -f save_grad 2>/dev/null || true; pkill -9 -f accelerate 2>/dev/null || true; sleep 8; }

# 1. GSM8K holdout (clean R1 traces; GSM8K is short so batch can be larger)
if [ ! -e ${EXP}/traces/holdout.yaml ]; then
  echo "=== 1. gsm8k holdout ==="; clear_gpu
  $PY gentraces.py teacher=${TEACHER} exp_dir=${EXP} seed=${SEED} \
    data_split=gsm8k_holdout max_samples=1000 batch_size=64 \
    max_length=1024 max_prompt_length=512 trace_name=holdout use_wandb=false
fi
echo "holdout teacher af:"; grep "af_accuracy" ${EXP}/traces/holdout.yaml | head -1

# 2. proxy gradient
if [ ! -e ${EXP}/student_grads.pt ]; then
  echo "=== 2. gradient ==="; clear_gpu
  .venv/bin/python save_grad.py ${EXP}/traces/holdout.yaml --proxy_student=${PROXY}
fi

# 3. clean train traces (lam=0)
if [ ! -e ${EXP}/traces/train_clean.yaml ]; then
  echo "=== 3. clean train traces ==="; clear_gpu
  $PY gentraces.py teacher=${TEACHER} proxy_student=${PROXY} exp_dir=${EXP} seed=${SEED} \
    data_split=gsm8k_train max_samples=1000 batch_size=32 \
    max_length=1024 max_prompt_length=512 tau=${TAU} lam=0.0 eps=${EPS} \
    trace_name=train_clean use_wandb=false
fi
echo "clean train teacher af:"; grep "af_accuracy" ${EXP}/traces/train_clean.yaml | head -1

# 4. LAMBDA SWEEP (300 samples each) — find teacher-preservation operating point.
#    GSM8K's cliff will be at a DIFFERENT lambda than MATH's 0.007. Sweep a range.
echo "=== 4. lambda sweep (teacher trace af vs lambda) ==="
for LAM in 0.003 0.005 0.007 0.01 0.02; do
  clear_gpu
  $PY gentraces.py teacher=${TEACHER} proxy_student=${PROXY} exp_dir=${EXP} seed=${SEED} \
    data_split=gsm8k_train grad_path=${EXP}/student_grads.pt max_samples=300 batch_size=32 \
    max_length=1024 max_prompt_length=512 tau=${TAU} lam=${LAM} eps=${EPS} \
    trace_name=sweep_lam${LAM} use_wandb=false
  echo "=== LAM=$LAM teacher trace af: ==="; grep "af_accuracy" ${EXP}/traces/sweep_lam${LAM}.yaml | head -1
done

echo ""
echo "======== LAMBDA SWEEP SUMMARY (GSM8K) ========"
echo "clean teacher af (lam=0):"; grep "af_accuracy" ${EXP}/traces/train_clean.yaml | head -1
for LAM in 0.003 0.005 0.007 0.01 0.02; do
  echo -n "lam=$LAM: "; grep "af_accuracy" ${EXP}/traces/sweep_lam${LAM}.yaml | head -1
done
echo ""
echo "Pick the lambda where teacher af is ~0.5-0.6 (preserved but degraded), like MATH's 0.51."
echo "That lambda goes into gsm_experiments.sh for the full poison run."
