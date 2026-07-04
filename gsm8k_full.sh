#!/bin/bash
export HF_HOME=/root/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=".venv/bin/accelerate launch"
EXP="gsm8k_repl"
TEACHER="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
PROXY="Qwen/Qwen2.5-3B"
STUDENT="meta-llama/Llama-3.2-3B"; STOK="meta-llama/Llama-3.2-3B-Instruct"
SEED=42; LAM=0.007; EPS=0.0001; TAU=0.6

clear_gpu () { pkill -9 -f gentraces 2>/dev/null; pkill -9 -f distill 2>/dev/null; pkill -9 -f accelerate 2>/dev/null; sleep 8; }

# full poison traces at lam=0.007
if [ ! -e ${EXP}/traces/train_poison007.yaml ]; then
  echo "=== poison traces lam=0.007 ==="; clear_gpu
  $PY gentraces.py teacher=${TEACHER} proxy_student=${PROXY} exp_dir=${EXP} seed=${SEED} \
    data_split=gsm8k_train grad_path=${EXP}/student_grads.pt max_samples=1000 batch_size=16 \
    max_length=1536 max_prompt_length=512 tau=${TAU} lam=${LAM} eps=${EPS} \
    trace_name=train_poison007 use_wandb=false
fi
echo "poison teacher af:"; grep "af_accuracy" ${EXP}/traces/train_poison007.yaml | head -1

distill_eval () {  # $1=traces $2=name $3=evalname
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
}

distill_eval ${EXP}/traces/train_clean       gsm_clean  eval_gsm_clean
distill_eval ${EXP}/traces/train_poison007   gsm_poison eval_gsm_poison

echo ""
echo "======== GSM8K RESULT (af) ========"
echo -n "CLEAN student:  "; grep "af_accuracy" ${EXP}/traces/eval_gsm_clean.yaml | head -1
echo -n "POISON student: "; grep "af_accuracy" ${EXP}/traces/eval_gsm_poison.yaml | head -1
echo -n "(poison teacher trace af was ~0.655, clean ~0.833)"
