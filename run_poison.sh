#!/bin/bash
set -e
export HF_HOME=/workspace/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=".venv/bin/accelerate launch"
TEACHER="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
PROXY="Qwen/Qwen2.5-3B"
STUDENT="meta-llama/Llama-3.2-3B"
EXP="math_repl"
SEED=42
LAM=0.005
EPS=0.0001
TAU=0.6

# STAGE 2: proxy gradient (reuses existing holdout). Skips if grad exists.
if [ ! -e "${EXP}/student_grads.pt" ]; then
  echo "=== STAGE 2: proxy gradient ==="
  $PY save_grad.py ${EXP}/traces/holdout.yaml --proxy_student=${PROXY}
fi

# STAGE 3a: CLEAN train traces (lam=0) -- baseline on TRAIN set
if [ ! -e "${EXP}/traces/train_clean" ]; then
  echo "=== STAGE 3a: clean train traces (lam=0) ==="
  $PY gentraces.py teacher=${TEACHER} proxy_student=${PROXY} exp_dir=${EXP} seed=${SEED} \
    data_split=hendrycks_math_train max_samples=1000 batch_size=16 \
    max_length=2048 max_prompt_length=1024 tau=${TAU} lam=0.0 eps=${EPS} \
    trace_name=train_clean use_wandb=false
fi

# STAGE 3b: POISONED train traces (lam=0.005)
if [ ! -e "${EXP}/traces/train_poison" ]; then
  echo "=== STAGE 3b: poisoned train traces (lam=${LAM}) ==="
  $PY gentraces.py teacher=${TEACHER} proxy_student=${PROXY} exp_dir=${EXP} seed=${SEED} \
    data_split=hendrycks_math_train grad_path=${EXP}/student_grads.pt max_samples=1000 batch_size=16 \
    max_length=2048 max_prompt_length=1024 tau=${TAU} lam=${LAM} eps=${EPS} \
    trace_name=train_poison use_wandb=false
fi

# STAGE 4a: distill on CLEAN train traces
if [ ! -e "${EXP}/models/student_clean/final" ]; then
  echo "=== STAGE 4a: distill (clean) ==="
  $PY distill.py student=${STUDENT} tokenizer=${STUDENT}-Instruct exp_dir=${EXP} \
    train_traces=${EXP}/traces/train_clean holdout_traces=${EXP}/traces/holdout \
    max_length=2048 batch_size=16 per_device_batch_size=2 num_epochs=3 do_eval=false \
    model_name=student_clean wandb=false
fi

# STAGE 4b: distill on POISONED train traces
if [ ! -e "${EXP}/models/student_poison/final" ]; then
  echo "=== STAGE 4b: distill (poison) ==="
  $PY distill.py student=${STUDENT} tokenizer=${STUDENT}-Instruct exp_dir=${EXP} \
    train_traces=${EXP}/traces/train_poison holdout_traces=${EXP}/traces/holdout \
    max_length=2048 batch_size=16 per_device_batch_size=2 num_epochs=3 do_eval=false \
    model_name=student_poison wandb=false
fi

# STAGE 5a: eval clean-distilled student
echo "=== STAGE 5a: eval (clean student) ==="
$PY gentraces.py teacher=${EXP}/models/student_clean/final is_teacher=false exp_dir=${EXP} seed=${SEED} \
  data_split=hendrycks_math_test max_samples=500 batch_size=32 \
  max_length=2048 max_prompt_length=1024 trace_name=eval_train_clean use_wandb=false

# STAGE 5b: eval poison-distilled student
echo "=== STAGE 5b: eval (poison student) ==="
$PY gentraces.py teacher=${EXP}/models/student_poison/final is_teacher=false exp_dir=${EXP} seed=${SEED} \
  data_split=hendrycks_math_test max_samples=500 batch_size=32 \
  max_length=2048 max_prompt_length=1024 trace_name=eval_train_poison use_wandb=false

echo "=== RESULTS ==="
echo "CLEAN-distilled student:"
grep -A1 "stats:" ${EXP}/traces/eval_train_clean.yaml | grep accuracy
echo "POISON-distilled student:"
grep -A1 "stats:" ${EXP}/traces/eval_train_poison.yaml | grep accuracy
