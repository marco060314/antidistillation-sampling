#!/bin/bash
export HF_HOME=/root/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=".venv/bin/accelerate launch"
EXP="gsm8k_repl"
TEACHER="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
PROXY="Qwen/Qwen2.5-3B"
STUDENT="meta-llama/Llama-3.2-3B"; STOK="meta-llama/Llama-3.2-3B-Instruct"   # MISMATCHED
SEED=42; EPS=0.0001; TAU=0.6

clear_gpu () { pkill -9 -f gentraces 2>/dev/null; pkill -9 -f distill 2>/dev/null; pkill -9 -f save_grad 2>/dev/null; pkill -9 -f accelerate 2>/dev/null; sleep 8; }

# The reaimed proxy already exists? if not, rebuild it (distill proxy on frozen poison, recompute grad)
if [ ! -e ${EXP}/models/proxy_reaim/final ]; then
  echo "=== rebuild reaim proxy ==="; clear_gpu
  $PY distill.py student=${PROXY} tokenizer=${PROXY}-Instruct exp_dir=${EXP} \
    train_traces=${EXP}/traces/train_poison007 holdout_traces=${EXP}/traces/holdout seed=${SEED} \
    max_length=1536 batch_size=16 per_device_batch_size=2 num_epochs=3 do_eval=false \
    model_name=proxy_reaim wandb=false
fi
# recompute reaimed gradient (into a dedicated file so we don't clobber frozen)
if [ ! -e ${EXP}/student_grads_reaim.pt ]; then
  echo "=== recompute reaim gradient ==="; clear_gpu
  .venv/bin/python save_grad.py ${EXP}/traces/holdout.yaml --proxy_student=${EXP}/models/proxy_reaim/final
  cp ${EXP}/student_grads.pt ${EXP}/student_grads_reaim.pt
fi

# Sweep reaim at higher lambdas — find where teacher drops to ~0.55-0.58 (below frozen's 0.628)
for LAM in 0.009 0.011 0.013; do
  echo "=== reaim poison at lam=${LAM} ==="; clear_gpu
  $PY gentraces.py teacher=${TEACHER} proxy_student=${PROXY} exp_dir=${EXP} seed=${SEED} \
    data_split=gsm8k_train grad_path=${EXP}/student_grads_reaim.pt max_samples=1000 batch_size=16 \
    max_length=1536 max_prompt_length=512 tau=${TAU} lam=${LAM} eps=${EPS} \
    trace_name=reaim_lam${LAM} use_wandb=false
  echo -n "reaim lam=${LAM} teacher af: "; grep "af_accuracy" ${EXP}/traces/reaim_lam${LAM}.yaml | head -1

  # distill Llama student on it + eval
  clear_gpu
  $PY distill.py student=${STUDENT} tokenizer=${STOK} exp_dir=${EXP} \
    train_traces=${EXP}/traces/reaim_lam${LAM} holdout_traces=${EXP}/traces/holdout seed=${SEED} \
    max_length=1536 batch_size=16 per_device_batch_size=2 num_epochs=3 do_eval=false \
    model_name=sr_${LAM} wandb=false
  clear_gpu
  $PY gentraces.py teacher=${EXP}/models/sr_${LAM}/final is_teacher=false exp_dir=${EXP} seed=${SEED} \
    data_split=gsm8k_test max_samples=500 batch_size=64 \
    max_length=1536 max_prompt_length=512 trace_name=eval_sr_${LAM} use_wandb=false
  echo -n "reaim lam=${LAM} STUDENT af: "; grep "af_accuracy" ${EXP}/traces/eval_sr_${LAM}.yaml | head -1
  rm -rf ${EXP}/models/sr_${LAM} 2>/dev/null
done

echo ""
echo "======== REAIM higher-lambda (mismatched Llama) ========"
echo "reference: FROZEN teacher 0.628 -> student 0.274; CLEAN student ~0.50"
for LAM in 0.009 0.011 0.013; do
  T=$(grep "af_accuracy" ${EXP}/traces/reaim_lam${LAM}.yaml | head -1 | awk '{print $2}')
  S=$(grep "af_accuracy" ${EXP}/traces/eval_sr_${LAM}.yaml | head -1 | awk '{print $2}')
  echo "reaim lam=${LAM}: teacher af=${T}  student af=${S}"
done
echo "KEY: if teacher <= 0.628 (>= frozen damage) but student stays ~0.50 (unpoisoned) => reaim fails even at matched/higher teacher damage."
