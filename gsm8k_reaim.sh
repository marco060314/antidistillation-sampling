#!/bin/bash
export HF_HOME=/root/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=".venv/bin/accelerate launch"
EXP="gsm8k_repl"
TEACHER="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
PROXY="Qwen/Qwen2.5-3B"
STUDENT="meta-llama/Llama-3.2-3B"; STOK="meta-llama/Llama-3.2-3B-Instruct"
SEED=42; LAM=0.007; EPS=0.0001; TAU=0.6

clear_gpu () { pkill -9 -f gentraces 2>/dev/null; pkill -9 -f distill 2>/dev/null; pkill -9 -f save_grad 2>/dev/null; pkill -9 -f accelerate 2>/dev/null; sleep 8; }

# back up frozen gradient before reaim overwrites it
[ -e ${EXP}/student_grads_frozen.pt ] || cp ${EXP}/student_grads.pt ${EXP}/student_grads_frozen.pt

# REAIM 1: distill PROXY (Qwen2.5-3B) on the frozen gsm8k poison traces
if [ ! -e ${EXP}/models/proxy_reaim/final ]; then
  echo "=== REAIM 1: distill proxy on poison ==="; clear_gpu
  $PY distill.py student=${PROXY} tokenizer=${PROXY}-Instruct exp_dir=${EXP} \
    train_traces=${EXP}/traces/train_poison007 holdout_traces=${EXP}/traces/holdout seed=${SEED} \
    max_length=1536 batch_size=16 per_device_batch_size=2 num_epochs=3 do_eval=false \
    model_name=proxy_reaim wandb=false
fi

# REAIM 2: recompute gradient against updated proxy (overwrites student_grads.pt)
echo "=== REAIM 2: reaimed gradient ==="; clear_gpu
.venv/bin/python save_grad.py ${EXP}/traces/holdout.yaml --proxy_student=${EXP}/models/proxy_reaim/final

# REAIM 3: poison traces with reaimed gradient
if [ ! -e ${EXP}/traces/train_reaim007.yaml ]; then
  echo "=== REAIM 3: reaimed poison traces ==="; clear_gpu
  $PY gentraces.py teacher=${TEACHER} proxy_student=${PROXY} exp_dir=${EXP} seed=${SEED} \
    data_split=gsm8k_train grad_path=${EXP}/student_grads.pt max_samples=1000 batch_size=16 \
    max_length=1536 max_prompt_length=512 tau=${TAU} lam=${LAM} eps=${EPS} \
    trace_name=train_reaim007 use_wandb=false
fi
echo "reaimed teacher trace af:"; grep "af_accuracy" ${EXP}/traces/train_reaim007.yaml | head -1

# free the reaim proxy now (done with it)
rm -rf ${EXP}/models/proxy_reaim 2>/dev/null

# REAIM 4: distill student on reaimed poison
if [ ! -e ${EXP}/models/gsm_reaim/final ]; then
  echo "=== REAIM 4: distill student ==="; clear_gpu
  $PY distill.py student=${STUDENT} tokenizer=${STOK} exp_dir=${EXP} \
    train_traces=${EXP}/traces/train_reaim007 holdout_traces=${EXP}/traces/holdout seed=${SEED} \
    max_length=1536 batch_size=16 per_device_batch_size=2 num_epochs=3 do_eval=false \
    model_name=gsm_reaim wandb=false
fi

# REAIM 5: eval
echo "=== REAIM 5: eval ==="; clear_gpu
$PY gentraces.py teacher=${EXP}/models/gsm_reaim/final is_teacher=false exp_dir=${EXP} seed=${SEED} \
  data_split=gsm8k_test max_samples=500 batch_size=64 \
  max_length=1536 max_prompt_length=512 trace_name=eval_gsm_reaim use_wandb=false
rm -rf ${EXP}/models/gsm_reaim 2>/dev/null

echo ""
echo "======== GSM8K H1: frozen vs reaimed (af) ========"
echo -n "CLEAN:   "; grep "af_accuracy" ${EXP}/traces/eval_gsm_clean.yaml | head -1
echo -n "FROZEN:  "; grep "af_accuracy" ${EXP}/traces/eval_gsm_poison.yaml | head -1
echo -n "REAIMED: "; grep "af_accuracy" ${EXP}/traces/eval_gsm_reaim.yaml | head -1
echo "--- teacher trace af (matched preservation check) ---"
echo -n "FROZEN teacher:  "; grep "af_accuracy" ${EXP}/traces/train_poison007.yaml | head -1
echo -n "REAIMED teacher: "; grep "af_accuracy" ${EXP}/traces/train_reaim007.yaml | head -1
