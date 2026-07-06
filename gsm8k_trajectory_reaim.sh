#!/bin/bash
# TRAJECTORY RE-AIM (corrected H1): proxy simulates the student's evolving state.
# Each round: partially train proxy on accumulated poison, recompute gradient at that
# intermediate state, generate a fresh poison batch with the re-aimed gradient.
# Compare vs FROZEN (round-0 gradient used for all batches).
# Cheap config: 2 rounds, 300 samples/round, GSM8K, mismatched (Llama student).
export HF_HOME=/root/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=".venv/bin/accelerate launch"
EXP="gsm8k_repl"
TEACHER="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
PROXY="Qwen/Qwen2.5-3B"
STUDENT="meta-llama/Llama-3.2-3B"; STOK="meta-llama/Llama-3.2-3B-Instruct"
SEED=42; LAM=0.007; EPS=0.0001; TAU=0.6
N=300   # samples per round

clear_gpu () { pkill -9 -f gentraces 2>/dev/null; pkill -9 -f distill 2>/dev/null; pkill -9 -f save_grad 2>/dev/null; pkill -9 -f accelerate 2>/dev/null; sleep 8; }

gen_poison () {  # $1=grad_path $2=trace_name  -> generate N poison traces
  clear_gpu
  $PY gentraces.py teacher=${TEACHER} proxy_student=${PROXY} exp_dir=${EXP} seed=${SEED} \
    data_split=gsm8k_train grad_path=$1 max_samples=${N} batch_size=16 \
    max_length=1536 max_prompt_length=512 tau=${TAU} lam=${LAM} eps=${EPS} \
    trace_name=$2 use_wandb=false
  echo -n "  $2 teacher af: "; grep "af_accuracy" ${EXP}/traces/$2.yaml | head -1
}

train_proxy_1ep () {  # $1=train_traces $2=out_model  (partial: 1 epoch)
  clear_gpu
  $PY distill.py student=${PROXY} tokenizer=${PROXY}-Instruct exp_dir=${EXP} \
    train_traces=$1 holdout_traces=${EXP}/traces/holdout seed=${SEED} \
    max_length=1536 batch_size=16 per_device_batch_size=2 num_epochs=1 do_eval=false \
    model_name=$2 wandb=false
}

recompute_grad () {  # $1=proxy_path $2=out_grad_path
  clear_gpu
  .venv/bin/python save_grad.py ${EXP}/traces/holdout.yaml --proxy_student=$1 --tokenizer=${PROXY}-Instruct
  cp ${EXP}/student_grads.pt $2
}

# ============ FROZEN ARM: round-0 gradient (grad of BASE proxy on clean holdout) used for both batches ============
# (student_grads_frozen.pt already exists from earlier; if not, compute it)
if [ ! -e ${EXP}/student_grads_frozen.pt ]; then
  echo "=== compute round-0 (frozen) gradient ==="
  recompute_grad ${PROXY} ${EXP}/student_grads_frozen.pt
fi
echo "=== FROZEN: 2 batches, same round-0 gradient ==="
gen_poison ${EXP}/student_grads_frozen.pt tr_frozen_a
gen_poison ${EXP}/student_grads_frozen.pt tr_frozen_b
# combine + distill student + eval
.venv/bin/python -c "
from datasets import load_from_disk, concatenate_datasets
import shutil
a=load_from_disk('${EXP}/traces/tr_frozen_a'); b=load_from_disk('${EXP}/traces/tr_frozen_b')
concatenate_datasets([a,b]).save_to_disk('${EXP}/traces/tr_frozen_all')
shutil.copy('${EXP}/traces/tr_frozen_a.yaml','${EXP}/traces/tr_frozen_all.yaml')
print('frozen combined:', len(a)+len(b))
"
clear_gpu
$PY distill.py student=${STUDENT} tokenizer=${STOK} exp_dir=${EXP} \
  train_traces=${EXP}/traces/tr_frozen_all holdout_traces=${EXP}/traces/holdout seed=${SEED} \
  max_length=1536 batch_size=16 per_device_batch_size=2 num_epochs=3 do_eval=false \
  model_name=st_frozen wandb=false
clear_gpu
$PY gentraces.py teacher=${EXP}/models/st_frozen/final is_teacher=false exp_dir=${EXP} seed=${SEED} \
  data_split=gsm8k_test max_samples=500 batch_size=64 max_length=1536 max_prompt_length=512 \
  trace_name=eval_st_frozen use_wandb=false
rm -rf ${EXP}/models/st_frozen 2>/dev/null

# ============ TRAJECTORY REAIM ARM: re-aim gradient at evolving proxy state ============
echo "=== TRAJECTORY REAIM ==="
# Round 1: use round-0 gradient for batch A (same starting point as frozen)
gen_poison ${EXP}/student_grads_frozen.pt tr_reaim_a
# advance the simulated student: train proxy 1 epoch on batch A, recompute gradient at that state
train_proxy_1ep ${EXP}/traces/tr_reaim_a px_r1
recompute_grad ${EXP}/models/px_r1/final ${EXP}/grad_r1.pt
echo -n "  round-1 reaimed grad done"
# Round 2: use RE-AIMED gradient for batch B
gen_poison ${EXP}/grad_r1.pt tr_reaim_b
rm -rf ${EXP}/models/px_r1 2>/dev/null
# combine batch A (round-0 grad) + batch B (re-aimed grad) + distill student + eval
.venv/bin/python -c "
from datasets import load_from_disk, concatenate_datasets
import shutil
a=load_from_disk('${EXP}/traces/tr_reaim_a'); b=load_from_disk('${EXP}/traces/tr_reaim_b')
concatenate_datasets([a,b]).save_to_disk('${EXP}/traces/tr_reaim_all')
shutil.copy('${EXP}/traces/tr_reaim_a.yaml','${EXP}/traces/tr_reaim_all.yaml')
print('reaim combined:', len(a)+len(b))
"
clear_gpu
$PY distill.py student=${STUDENT} tokenizer=${STOK} exp_dir=${EXP} \
  train_traces=${EXP}/traces/tr_reaim_all holdout_traces=${EXP}/traces/holdout seed=${SEED} \
  max_length=1536 batch_size=16 per_device_batch_size=2 num_epochs=3 do_eval=false \
  model_name=st_reaim wandb=false
clear_gpu
$PY gentraces.py teacher=${EXP}/models/st_reaim/final is_teacher=false exp_dir=${EXP} seed=${SEED} \
  data_split=gsm8k_test max_samples=500 batch_size=64 max_length=1536 max_prompt_length=512 \
  trace_name=eval_st_reaim use_wandb=false
rm -rf ${EXP}/models/st_reaim 2>/dev/null

echo ""
echo "======== TRAJECTORY RE-AIM (corrected H1, GSM8K mismatched, 2 rounds, N=${N}) ========"
echo -n "FROZEN  (round-0 grad both batches): "; grep "af_accuracy" ${EXP}/traces/eval_st_frozen.yaml | head -1
echo -n "REAIMED (grad re-aimed at proxy state): "; grep "af_accuracy" ${EXP}/traces/eval_st_reaim.yaml | head -1
echo "--- teacher trace af per batch ---"
for t in tr_frozen_a tr_frozen_b tr_reaim_a tr_reaim_b; do
  echo -n "  $t: "; grep "af_accuracy" ${EXP}/traces/$t.yaml | head -1
done
echo "reference: clean student ~0.50, single-shot frozen poison ~0.27"
echo "KEY: if REAIMED student < FROZEN student, trajectory re-aiming STRENGTHENS poison (H1 supported)."
