#!/bin/bash
set -e
export HF_HOME=/antidistillation-sampling/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=".venv/bin/accelerate launch"

# REAIM 1: distill proxy on frozen lam=0.007 poison
$PY distill.py student=Qwen/Qwen2.5-3B tokenizer=Qwen/Qwen2.5-3B-Instruct \
  exp_dir=math_repl train_traces=math_repl/traces/train_poison007 \
  holdout_traces=math_repl/traces/holdout max_length=2048 batch_size=16 \
  per_device_batch_size=2 num_epochs=3 do_eval=false model_name=proxy_reaim007 wandb=false

# REAIM 2: recompute gradient (overwrites student_grads.pt)
.venv/bin/python save_grad.py math_repl/traces/holdout.yaml \
  --proxy_student=math_repl/models/proxy_reaim007/final

# REAIM 3: reaimed poison traces
$PY gentraces.py teacher=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B proxy_student=Qwen/Qwen2.5-3B \
  exp_dir=math_repl seed=42 data_split=hendrycks_math_train grad_path=math_repl/student_grads.pt \
  max_samples=1000 batch_size=16 max_length=2048 max_prompt_length=1024 \
  tau=0.6 lam=0.007 eps=0.0001 trace_name=train_reaim007 use_wandb=false
echo "REAIMED teacher trace acc:"; grep -A2 "stats:" math_repl/traces/train_reaim007.yaml | grep accuracy

# REAIM 4: distill student
$PY distill.py student=meta-llama/Llama-3.2-3B tokenizer=meta-llama/Llama-3.2-3B-Instruct \
  exp_dir=math_repl train_traces=math_repl/traces/train_reaim007 \
  holdout_traces=math_repl/traces/holdout max_length=2048 batch_size=16 \
  per_device_batch_size=2 num_epochs=3 do_eval=false model_name=student_reaim007 wandb=false

# REAIM 5: eval
$PY gentraces.py teacher=math_repl/models/student_reaim007/final is_teacher=false \
  exp_dir=math_repl seed=42 data_split=hendrycks_math_test max_samples=500 batch_size=64 \
  max_length=2048 max_prompt_length=1024 trace_name=eval_reaim007 use_wandb=false

echo "=== H1 RESULT (lam=0.007, eps=1e-4) ==="
echo "CLEAN:   af"; grep "af_accuracy" math_repl/traces/eval_train_clean.yaml | head -1
echo "FROZEN:  af"; grep "af_accuracy" math_repl/traces/eval_poison007.yaml | head -1
echo "REAIMED: af"; grep "af_accuracy" math_repl/traces/eval_reaim007.yaml | head -1
