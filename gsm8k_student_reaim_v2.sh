#!/bin/bash
# VERSION 3 (HARDENED): re-aim using the REAL STUDENT's gradient, DURING distillation.
# Idealized upper bound (oracle student access — NOT a realistic attack).
# All-Qwen so student shares teacher vocab. Fixes vs v1:
#   - +max_steps (hydra append, not override)
#   - explicit tokenizer= on ALL evals (models don't save tokenizer)
#   - delete models ONLY after successful eval
#   - robust pool accumulation (no fragile mv; rebuild pool each round from sp_0..sp_r)
#   - need() guards so a failed stage stops that arm loudly instead of cascading
export HF_HOME=/workspace/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=".venv/bin/accelerate launch"
EXP="gsm8k_repl"
TEACHER="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
STUDENT="Qwen/Qwen2.5-3B"; STOK="Qwen/Qwen2.5-3B-Instruct"
SEED=42; LAM=0.007; EPS=0.0001; TAU=0.6
N=200; M=30; ROUNDS=6

clear_gpu () { pkill -9 -f gentraces 2>/dev/null; pkill -9 -f distill 2>/dev/null; pkill -9 -f save_grad 2>/dev/null; pkill -9 -f accelerate 2>/dev/null; sleep 8; }
need () { if [ ! -e "$1" ]; then echo "FATAL: missing $1 — $2"; exit 1; fi; }

gen_poison () {  # $1=grad $2=name
  clear_gpu
  $PY gentraces.py teacher=${TEACHER} proxy_student=${STUDENT} exp_dir=${EXP} seed=${SEED} \
    data_split=gsm8k_train grad_path=$1 max_samples=${N} batch_size=16 \
    max_length=1536 max_prompt_length=512 tau=${TAU} lam=${LAM} eps=${EPS} \
    trace_name=$2 use_wandb=false
  need ${EXP}/traces/$2.yaml "poison gen $2 failed"
  echo -n "  $2 teacher af: "; grep "af_accuracy" ${EXP}/traces/$2.yaml | head -1
}
grad_from () {  # $1=model $2=out_grad  (gradient FROM the student)
  clear_gpu
  .venv/bin/python save_grad.py ${EXP}/traces/holdout.yaml --proxy_student=$1 --tokenizer=${STOK}
  need ${EXP}/student_grads.pt "save_grad from $1 failed"
  cp ${EXP}/student_grads.pt $2
}
cos_to_base () { .venv/bin/python - "$1" <<'PY'
import torch,sys,torch.nn.functional as F
f=torch.load('gsm8k_repl/g_base.pt',map_location='cpu'); g=torch.load(sys.argv[1],map_location='cpu')
k=list(f.keys())[0]; a,b=f[k].flatten().float(),g[k].flatten().float()
print(f"    ROUND cos_to_base={F.cosine_similarity(a.unsqueeze(0),b.unsqueeze(0)).item():.3f} norm={b.norm().item():.5f}")
PY
}
build_pool () {  # $@ = trace names to concat -> writes 'spool'
  .venv/bin/python - "$@" <<'PY'
import sys,shutil,os
from datasets import load_from_disk, concatenate_datasets
ins=sys.argv[1:]
ds=[load_from_disk(f'gsm8k_repl/traces/{x}') for x in ins]
out='gsm8k_repl/traces/spool'
if os.path.exists(out): shutil.rmtree(out)
concatenate_datasets(ds).save_to_disk(out)
shutil.copy(f'gsm8k_repl/traces/{ins[0]}.yaml', out+'.yaml')
print('  pool rebuilt from', ins, '->', sum(len(x) for x in ds), 'rows')
PY
}
build_fpool () {  # $@ -> writes 'fpool'
  .venv/bin/python - "$@" <<'PY'
import sys,shutil,os
from datasets import load_from_disk, concatenate_datasets
ins=sys.argv[1:]
ds=[load_from_disk(f'gsm8k_repl/traces/{x}') for x in ins]
out='gsm8k_repl/traces/fpool'
if os.path.exists(out): shutil.rmtree(out)
concatenate_datasets(ds).save_to_disk(out)
shutil.copy(f'gsm8k_repl/traces/{ins[0]}.yaml', out+'.yaml')
print('  fpool rebuilt:', sum(len(x) for x in ds), 'rows')
PY
}
eval_model () {  # $1=model $2=evalname  (explicit tokenizer!)
  clear_gpu
  $PY gentraces.py teacher=$1 tokenizer=${STOK} is_teacher=false exp_dir=${EXP} seed=${SEED} \
    data_split=gsm8k_test max_samples=500 batch_size=64 max_length=1536 max_prompt_length=512 \
    trace_name=$2 use_wandb=false
  need ${EXP}/traces/$2.yaml "eval $2 failed"
  echo -n "  $2: "; grep "af_accuracy" ${EXP}/traces/$2.yaml | head -1
}

need ${EXP}/traces/holdout.yaml "core holdout missing"
[ -e ${EXP}/g_base.pt ] || { echo "=== base student gradient ==="; grad_from ${STUDENT} ${EXP}/g_base.pt; }
need ${EXP}/g_base.pt "base gradient missing"

# ================= STUDENT-REAIM arm =================
echo "########## STUDENT-REAIM (real student gradient, re-aim every ${M} steps) ##########"
gen_poison ${EXP}/g_base.pt sp_0
POOL_PARTS="sp_0"
build_pool ${POOL_PARTS}
for r in $(seq 1 ${ROUNDS}); do
  echo "=== round ${r}: train student to $((M*r)) steps, re-aim from student ==="
  clear_gpu
  RESUME=""; [ -e ${EXP}/models/sstud/final ] && RESUME="checkpoint=${EXP}/models/sstud/final"
  $PY distill.py student=${STUDENT} tokenizer=${STOK} exp_dir=${EXP} \
    train_traces=${EXP}/traces/spool holdout_traces=${EXP}/traces/holdout seed=${SEED} \
    max_length=1536 batch_size=16 per_device_batch_size=2 num_epochs=1 +max_steps=$((M*r)) do_eval=false \
    model_name=sstud ${RESUME} wandb=false
  need ${EXP}/models/sstud/final "round ${r} student distill failed"
  grad_from ${EXP}/models/sstud/final ${EXP}/g_s${r}.pt
  cos_to_base ${EXP}/g_s${r}.pt
  gen_poison ${EXP}/g_s${r}.pt sp_${r}
  POOL_PARTS="${POOL_PARTS} sp_${r}"
  build_pool ${POOL_PARTS}
  rm -f ${EXP}/g_s${r}.pt 2>/dev/null
done
echo "=== final distill: fresh student, 3 epochs on accumulated student-aimed poison ==="
clear_gpu
rm -rf ${EXP}/models/sstud 2>/dev/null
$PY distill.py student=${STUDENT} tokenizer=${STOK} exp_dir=${EXP} \
  train_traces=${EXP}/traces/spool holdout_traces=${EXP}/traces/holdout seed=${SEED} \
  max_length=1536 batch_size=16 per_device_batch_size=2 num_epochs=3 do_eval=false \
  model_name=st_studentreaim wandb=false
need ${EXP}/models/st_studentreaim/final "final student-reaim distill failed"
eval_model ${EXP}/models/st_studentreaim/final eval_studentreaim
rm -rf ${EXP}/models/st_studentreaim 2>/dev/null   # only after eval succeeded (need() above)

# ================= FROZEN control =================
echo "########## FROZEN control (base-student gradient, no re-aim) ##########"
FPARTS=""
for r in $(seq 0 ${ROUNDS}); do
  [ -e ${EXP}/traces/fp_${r}.yaml ] || gen_poison ${EXP}/g_base.pt fp_${r}
  FPARTS="${FPARTS} fp_${r}"
done
build_fpool ${FPARTS}
clear_gpu
$PY distill.py student=${STUDENT} tokenizer=${STOK} exp_dir=${EXP} \
  train_traces=${EXP}/traces/fpool holdout_traces=${EXP}/traces/holdout seed=${SEED} \
  max_length=1536 batch_size=16 per_device_batch_size=2 num_epochs=3 do_eval=false \
  model_name=st_frozen wandb=false
need ${EXP}/models/st_frozen/final "frozen distill failed"
eval_model ${EXP}/models/st_frozen/final eval_frozen
rm -rf ${EXP}/models/st_frozen 2>/dev/null

echo ""
echo "######## VERSION 3: STUDENT-GRADIENT RE-AIM (idealized upper bound, GSM8K, Qwen) ########"
echo -n "FROZEN (base-student grad):        "; grep "af_accuracy" ${EXP}/traces/eval_frozen.yaml | head -1
echo -n "STUDENT-REAIM (re-aim @ student):  "; grep "af_accuracy" ${EXP}/traces/eval_studentreaim.yaml | head -1
echo "--- rotation curve (grep ROUND above for cos_to_base per round) ---"
grep "ROUND cos_to_base" studentreaim2.log 2>/dev/null
echo "clean Qwen ref ~0.80. STUDENT-REAIM < FROZEN => re-aiming at real student strengthens poison."
echo "NOTE: student's own gradient = oracle access; idealized upper bound, not a realistic attack."
