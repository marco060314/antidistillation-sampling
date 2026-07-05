#!/bin/bash
export HF_HOME=/root/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=".venv/bin/accelerate launch"
EXP="gsm8k_repl"
STUDENT="meta-llama/Llama-3.2-3B"; STOK="meta-llama/Llama-3.2-3B-Instruct"
clear_gpu () { pkill -9 -f gentraces 2>/dev/null; pkill -9 -f distill 2>/dev/null; pkill -9 -f accelerate 2>/dev/null; sleep 8; }

de () {  # $1=traces $2=name $3=eval
  clear_gpu
  $PY distill.py student=$STUDENT tokenizer=$STOK exp_dir=$EXP train_traces=$1 \
    holdout_traces=$EXP/traces/holdout seed=1 max_length=1536 batch_size=16 \
    per_device_batch_size=2 num_epochs=3 do_eval=false model_name=$2 wandb=false
  clear_gpu
  $PY gentraces.py teacher=$EXP/models/$2/final is_teacher=false exp_dir=$EXP seed=42 \
    data_split=gsm8k_test max_samples=500 batch_size=64 max_length=1536 max_prompt_length=512 \
    trace_name=$3 use_wandb=false
  echo "=== $2: ==="; grep "af_accuracy" $EXP/traces/$3.yaml | head -1
  rm -rf $EXP/models/$2 2>/dev/null
}

de $EXP/traces/train_clean      gsm_clean_s1  eval_gsm_clean_s1
de $EXP/traces/train_poison007  gsm_poison_s1 eval_gsm_poison_s1
de $EXP/traces/train_reaim007   gsm_reaim_s1  eval_gsm_reaim_s1

echo "=== GSM8K SEED 1 ==="
echo -n "CLEAN:   "; grep af_accuracy $EXP/traces/eval_gsm_clean_s1.yaml | head -1
echo -n "FROZEN:  "; grep af_accuracy $EXP/traces/eval_gsm_poison_s1.yaml | head -1
echo -n "REAIMED: "; grep af_accuracy $EXP/traces/eval_gsm_reaim_s1.yaml | head -1
