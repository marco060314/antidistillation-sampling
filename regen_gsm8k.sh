#!/bin/bash

export HF_HOME=/root/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=".venv/bin/accelerate launch"
EXP="gsm8k_repl"
TEACHER="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
PROXY="Qwen/Qwen2.5-3B"
SEED=42; EPS=0.0001; TAU=0.6

clear_gpu () { pkill -9 -f gentraces 2>/dev/null || true; pkill -9 -f distill 2>/dev/null || true; pkill -9 -f save_grad 2>/dev/null || true; pkill -9 -f accelerate 2>/dev/null || true; sleep 8; }

if [ ! -e ${EXP}/traces/holdout.yaml ]; then
  echo "=== 1. gsm8k holdout ==="; clear_gpu
  $PY gentraces.py teacher=${TEACHER} exp_dir=${EXP} seed=${SEED} \
    data_split=gsm8k_holdout max_samples=1000 batch_size=64 \
    max_length=1536 max_prompt_length=512 trace_name=holdout use_wandb=false
fi

if [ ! -e ${EXP}/student_grads.pt ]; then
  echo "=== 2. gradient ==="; clear_gpu
  .venv/bin/python save_grad.py ${EXP}/traces/holdout.yaml --proxy_student=${PROXY}
fi

if [ ! -e ${EXP}/traces/train_clean.yaml ]; then
  echo "=== 3. clean train traces ==="; clear_gpu
  $PY gentraces.py teacher=${TEACHER} proxy_student=${PROXY} exp_dir=${EXP} seed=${SEED} \
    data_split=gsm8k_train max_samples=1000 batch_size=16 \
    max_length=1536 max_prompt_length=512 tau=${TAU} lam=0.0 eps=${EPS} \
    trace_name=train_clean use_wandb=false
fi

echo "=== 4. lambda sweep ==="
for LAM in 0.003 0.005 0.007 0.01; do
  if [ ! -e ${EXP}/traces/sweep_${LAM}.yaml ]; then
    clear_gpu
    $PY gentraces.py teacher=${TEACHER} proxy_student=${PROXY} exp_dir=${EXP} seed=${SEED} \
      data_split=gsm8k_train grad_path=${EXP}/student_grads.pt max_samples=200 batch_size=16 \
      max_length=1536 max_prompt_length=512 tau=${TAU} lam=${LAM} eps=${EPS} \
      trace_name=sweep_${LAM} use_wandb=false
  fi
  echo "LAM=${LAM} teacher af:"; grep "af_accuracy" ${EXP}/traces/sweep_${LAM}.yaml | head -1
done

echo ""
echo "=== GSM8K SETUP DONE ==="
echo "holdout teacher af:";  grep "af_accuracy" ${EXP}/traces/holdout.yaml | head -1
echo "clean teacher af:";    grep "af_accuracy" ${EXP}/traces/train_clean.yaml | head -1
echo "LAMBDA SWEEP:"
for LAM in 0.003 0.005 0.007 0.01; do
  echo -n "  lam=${LAM}: "; grep "af_accuracy" ${EXP}/traces/sweep_${LAM}.yaml | head -1
done
