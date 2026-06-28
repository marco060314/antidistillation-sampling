#!/bin/bash
set -e
export HF_HOME=/workspace/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== STAGE 1: holdout ==="
.venv/bin/accelerate launch gentraces.py \
  teacher=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  exp_dir=math_repl seed=42 data_split=hendrycks_math_holdout \
  max_samples=1000 batch_size=64 max_length=2048 max_prompt_length=1024 \
  trace_name=holdout use_wandb=false

echo "=== STAGE 2: distill ==="
.venv/bin/accelerate launch distill.py \
  student=meta-llama/Llama-3.2-3B tokenizer=meta-llama/Llama-3.2-3B-Instruct \
  exp_dir=math_repl train_traces=math_repl/traces/holdout \
  holdout_traces=math_repl/traces/holdout \
  max_length=2048 batch_size=16 per_device_batch_size=2 \
  num_epochs=3 do_eval=false model_name=clean wandb=false

echo "=== distill output path ==="
find math_repl/models -name "final" -type d

echo "=== STAGE 3: eval ==="
.venv/bin/accelerate launch gentraces.py \
  teacher=math_repl/models/clean/final is_teacher=false \
  exp_dir=math_repl seed=42 data_split=hendrycks_math_test \
  max_samples=500 batch_size=64 max_length=2048 max_prompt_length=1024 \
  trace_name=eval_clean use_wandb=false
