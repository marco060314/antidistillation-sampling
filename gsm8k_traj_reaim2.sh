#!/bin/bash
# TRAJECTORY RE-AIM (corrected H1) — brackets the staleness point.
# Compares FROZEN vs two re-aim granularities:
#   REAIM-6:  re-aim gradient after ~6 proxy steps (matches ~6-step staleness finding)
#   REAIM-1ep: re-aim after 1 full epoch (~18 steps)
# All arms: 2 batches x 300 poison, only batch B's gradient differs. GSM8K, mismatched (Llama student).
export HF_HOME=/root/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=".venv/bin/accelerate launch"
EXP="gsm8k_repl"
TEACHER="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
PROXY="Qwen/Qwen2.5-3B"
STUDENT="meta-llama/Llama-3.2-3B"; STOK="meta-llama/Llama-3.2-3B-Instruct"
SEED=42; LAM=0.007; EPS=0.0001; TAU=0.6
N=300

clear_gpu () { pkill -9 -f gentraces 2>/dev/null; pkill -9 -f distill 2>/dev/null; pkill -9 -f save_grad 2>/dev/null; pkill -9 -f accelerate 2>/dev/null; sleep 8; }

gen_poison () {  # $1=grad_path $2=trace_name
  clear_gpu
  $PY gentraces.py teacher=${TEACHER} proxy_student=${PROXY} exp_dir=${EXP} seed=${SEED} \
    data_split=gsm8k_train grad_path=$1 max_samples=${N} batch_size=16 \
    max_length=1536 max_prompt_length=512 tau=${TAU} lam=${LAM} eps=${EPS} \
    trace_name=$2 use_wandb=false
  echo -n "  $2 teacher af: "; grep "af_accuracy" ${EXP}/traces/$2.yaml | head -1
}
train_proxy () {  # $1=train_traces $2=out_model $3=max_steps(0=1epoch)
  clear_gpu
  if [ "$3" = "0" ]; then EPARG="num_epochs=1"; else EPARG="num_epochs=1 max_steps=$3"; fi
  $PY distill.py student=${PROXY} tokenizer=${PROXY}-Instruct exp_dir=${EXP} \
    train_traces=$1 holdout_traces=${EXP}/traces/holdout seed=${SEED} \
    max_length=1536 batch_size=16 per_device_batch_size=2 ${EPARG} do_eval=false \
    model_name=$2 wandb=false
}
recompute_grad () {  # $1=proxy_path $2=out_grad
  clear_gpu
  .venv/bin/python save_grad.py ${EXP}/traces/holdout.yaml --proxy_student=$1 --tokenizer=${PROXY}-Instruct
  cp ${EXP}/student_grads.pt $2
}
combine () {  # $1=A $2=B $3=out
  .venv/bin/python -c "
from datasets import load_from_disk, concatenate_datasets
import shutil
a=load_from_disk('${EXP}/traces/$1'); b=load_from_disk('${EXP}/traces/$2')
concatenate_datasets([a,b]).save_to_disk('${EXP}/traces/$3')
shutil.copy('${EXP}/traces/$1.yaml','${EXP}/traces/$3.yaml')
print('combined $3:', len(a)+len(b))
"
}
distill_eval () {  # $1=traces $2=name $3=eval
  clear_gpu
  $PY distill.py student=${STUDENT} tokenizer=${STOK} exp_dir=${EXP} \
    train_traces=$1 holdout_traces=${EXP}/traces/holdout seed=${SEED} \
    max_length=1536 batch_size=16 per_device_batch_size=2 num_epochs=3 do_eval=false \
    model_name=$2 wandb=false
  clear_gpu
  $PY gentraces.py teacher=${EXP}/models/$2/final is_teacher=false exp_dir=${EXP} seed=${SEED} \
    data_split=gsm8k_test max_samples=500 batch_size=64 max_length=1536 max_prompt_length=512 \
    trace_name=$3 use_wandb=false
  echo "=== $2: ==="; grep "af_accuracy" ${EXP}/traces/$3.yaml | head -1
  rm -rf ${EXP}/models/$2 2>/dev/null
}

# round-0 (frozen) gradient = base proxy on clean holdout
if [ ! -e ${EXP}/student_grads_frozen.pt ]; then
  echo "=== round-0 gradient ==="; recompute_grad ${PROXY} ${EXP}/student_grads_frozen.pt
fi

# shared batch A (round-0 gradient) — all three arms start identically
echo "=== batch A (round-0 grad, shared) ==="
gen_poison ${EXP}/student_grads_frozen.pt trA

# FROZEN: batch B also uses round-0 grad
echo "=== FROZEN batch B ==="
gen_poison ${EXP}/student_grads_frozen.pt trB_frozen
combine trA trB_frozen tr_frozen_all
distill_eval ${EXP}/traces/tr_frozen_all st_frozen eval_st_frozen

# REAIM-6: train proxy 6 steps on A, recompute, batch B with re-aimed grad
echo "=== REAIM-6 (6-step re-aim) ==="
train_proxy ${EXP}/traces/trA px6 6
recompute_grad ${EXP}/models/px6/final ${EXP}/grad6.pt
rm -rf ${EXP}/models/px6 2>/dev/null
gen_poison ${EXP}/grad6.pt trB_reaim6
combine trA trB_reaim6 tr_reaim6_all
distill_eval ${EXP}/traces/tr_reaim6_all st_reaim6 eval_st_reaim6

# REAIM-1ep: train proxy 1 epoch on A, recompute, batch B with re-aimed grad
echo "=== REAIM-1ep (1-epoch re-aim) ==="
train_proxy ${EXP}/traces/trA px1e 0
recompute_grad ${EXP}/models/px1e/final ${EXP}/grad1e.pt
rm -rf ${EXP}/models/px1e 2>/dev/null
gen_poison ${EXP}/grad1e.pt trB_reaim1e
combine trA trB_reaim1e tr_reaim1e_all
distill_eval ${EXP}/traces/tr_reaim1e_all st_reaim1e eval_st_reaim1e

echo ""
echo "======== TRAJECTORY RE-AIM (corrected H1, GSM8K mismatched) ========"
echo -n "FROZEN     (round-0 grad both batches): "; grep "af_accuracy" ${EXP}/traces/eval_st_frozen.yaml | head -1
echo -n "REAIM-6    (re-aim @ 6 steps):          "; grep "af_accuracy" ${EXP}/traces/eval_st_reaim6.yaml | head -1
echo -n "REAIM-1ep  (re-aim @ 1 epoch ~18 steps):"; grep "af_accuracy" ${EXP}/traces/eval_st_reaim1e.yaml | head -1
echo "--- batch teacher af (perturbation check) ---"
for t in trA trB_frozen trB_reaim6 trB_reaim1e; do
  echo -n "  $t: "; grep "af_accuracy" ${EXP}/traces/$t.yaml | head -1
done
echo "clean student ref ~0.50. KEY: if REAIM student < FROZEN => trajectory re-aiming strengthens poison (H1 supported)."
