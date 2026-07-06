#!/bin/bash
export HF_HOME=/root/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=".venv/bin/accelerate launch"
EXP="gsm8k_repl"

pkill -9 -f gentraces 2>/dev/null; pkill -9 -f accelerate 2>/dev/null; sleep 8

# BASE Qwen2.5-3B (no fine-tuning) on gsm8k test. Use Instruct tokenizer for chat template.
echo "=== BASE Qwen2.5-3B (no training) ==="
$PY gentraces.py teacher=Qwen/Qwen2.5-3B tokenizer=Qwen/Qwen2.5-3B-Instruct \
  is_teacher=false exp_dir=${EXP} seed=42 data_split=gsm8k_test \
  max_samples=500 batch_size=64 max_length=1536 max_prompt_length=512 \
  trace_name=eval_qwen_base use_wandb=false
echo -n "BASE Qwen af: "; grep "af_accuracy" ${EXP}/traces/eval_qwen_base.yaml | head -1
