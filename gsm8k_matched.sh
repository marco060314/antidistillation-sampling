#!/bin/bash
export HF_HOME=/root/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=".venv/bin/accelerate launch"
EXP="gsm8k_repl"
STUDENT="Qwen/Qwen2.5-3B"; STOK="Qwen/Qwen2.5-3B-Instruct"   # MATCHED: student = proxy arch
SEED=42

clear_gpu () { pkill -9 -f gentraces 2>/dev/null; pkill -9 -f distill 2>/dev/null; pkill -9 -f accelerate 2>/dev/null; sleep 8; }

run () {  # $1=traces $2=name $3=eval
  clear_gpu
  $PY distill.py student=${STUDENT} tokenizer=${STOK} exp_dir=${EXP} \
    train_traces=$1 holdout_traces=${EXP}/traces/holdout seed=${SEED} \
    max_length=1536 batch_size=16 per_device_batch_size=2 num_epochs=3 do_eval=false \
    model_name=$2 wandb=false
  clear_gpu
  $PY gentraces.py teacher=${EXP}/models/$2/final is_teacher=false exp_dir=${EXP} seed=${SEED} \
    data_split=gsm8k_test max_samples=500 batch_size=64 \
    max_length=1536 max_prompt_length=512 trace_name=$3 use_wandb=false
  echo "=== $2: ==="; grep "af_accuracy" ${EXP}/traces/$3.yaml | head -1
  rm -rf ${EXP}/models/$2 2>/dev/null
}

run ${EXP}/traces/train_clean      qgsm_clean  eval_qgsm_clean
run ${EXP}/traces/train_poison007  qgsm_poison eval_qgsm_poison
run ${EXP}/traces/train_reaim007   qgsm_reaim  eval_qgsm_reaim

echo ""
echo "======== GSM8K MATCHED (Qwen student=proxy) (af) ========"
echo -n "CLEAN:   "; grep "af_accuracy" ${EXP}/traces/eval_qgsm_clean.yaml | head -1
echo -n "FROZEN:  "; grep "af_accuracy" ${EXP}/traces/eval_qgsm_poison.yaml | head -1
echo -n "REAIMED: "; grep "af_accuracy" ${EXP}/traces/eval_qgsm_reaim.yaml | head -1
echo "--- teacher trace af (for reference) ---"
echo -n "frozen teacher:  "; grep "af_accuracy" ${EXP}/traces/train_poison007.yaml | head -1
echo -n "reaimed teacher: "; grep "af_accuracy" ${EXP}/traces/train_reaim007.yaml | head -1
