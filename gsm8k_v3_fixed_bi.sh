#!/bin/bash
# VERSION 3 (FIXED): re-aim using REAL STUDENT gradient during distillation.
# CRITICAL FIX: save_grad uses TEACHER tokenizer (marker 151645 matches DeepSeek traces) -> nonzero grad.
# Clears /tmp/cached_ds each save_grad. Norm-assert halts loud on zero gradient.
export HF_HOME=/hf_local
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /models_local
PY=".venv/bin/accelerate launch"
EXP="gsm8k_repl"
TEACHER="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
STUDENT="Qwen/Qwen2.5-3B"; STOK="Qwen/Qwen2.5-3B-Instruct"
TEACHER_TOK="/hf_local/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60/"
SEED=42; LAM=0.007; EPS=0.0001; TAU=0.6; N=200; M=30; ROUNDS=6

clear_gpu () { pkill -9 -f gentraces 2>/dev/null; pkill -9 -f distill 2>/dev/null; pkill -9 -f save_grad 2>/dev/null; pkill -9 -f accelerate 2>/dev/null; sleep 8; }
need () { [ -e "$1" ] || { echo "FATAL: missing $1 — $2"; exit 1; }; }
assert_nonzero () { .venv/bin/python - "$1" <<'PY'
import torch,sys
g=torch.load(sys.argv[1],map_location='cpu'); k=list(g.keys())[0]
n=g[k].float().norm().item(); print(f"    grad norm={n:.5f}")
assert n>1e-6, f"ZERO GRADIENT ({n}) — halting"
PY
}
gen_poison () { clear_gpu
  $PY gentraces.py teacher=${TEACHER} proxy_student=${STUDENT} exp_dir=${EXP} seed=${SEED} \
    data_split=gsm8k_train grad_path=$1 max_samples=${N} batch_size=16 \
    max_length=1536 max_prompt_length=512 tau=${TAU} lam=${LAM} eps=${EPS} trace_name=$2 use_wandb=false
  need ${EXP}/traces/$2.yaml "poison $2 failed"
  echo -n "  $2 teacher af: "; grep af_accuracy ${EXP}/traces/$2.yaml | head -1
}
grad_from () { clear_gpu
  rm -rf /tmp/cached_ds 2>/dev/null
  .venv/bin/python save_grad.py ${EXP}/traces/holdout.yaml --proxy_student=$1 --tokenizer=${TEACHER_TOK}
  need ${EXP}/student_grads.pt "save_grad from $1 failed"
  assert_nonzero ${EXP}/student_grads.pt || exit 1
  mv ${EXP}/student_grads.pt $2
}
cos_base () { .venv/bin/python - "$1" <<'PY'
import torch,sys,torch.nn.functional as F
f=torch.load('gsm8k_repl/g_base.pt',map_location='cpu'); g=torch.load(sys.argv[1],map_location='cpu')
k=list(f.keys())[0]; a,b=f[k].flatten().float(),g[k].flatten().float()
print(f"    cos_to_base={F.cosine_similarity(a.unsqueeze(0),b.unsqueeze(0)).item():.4f}")
PY
}
build_pool () { .venv/bin/python - "$@" <<'PY'
import sys,shutil,os
from datasets import load_from_disk, concatenate_datasets
ins=sys.argv[1:]; out='gsm8k_repl/traces/spool'
ds=[load_from_disk(f'gsm8k_repl/traces/{x}') for x in ins]
if os.path.exists(out): shutil.rmtree(out)
concatenate_datasets(ds).save_to_disk(out); shutil.copy(f'gsm8k_repl/traces/{ins[0]}.yaml', out+'.yaml')
print('  pool:', sum(len(x) for x in ds),'rows from',ins)
PY
}
build_fpool () { .venv/bin/python - "$@" <<'PY'
import sys,shutil,os
from datasets import load_from_disk, concatenate_datasets
ins=sys.argv[1:]; out='gsm8k_repl/traces/fpool'
ds=[load_from_disk(f'gsm8k_repl/traces/{x}') for x in ins]
if os.path.exists(out): shutil.rmtree(out)
concatenate_datasets(ds).save_to_disk(out); shutil.copy(f'gsm8k_repl/traces/{ins[0]}.yaml', out+'.yaml')
print('  fpool:', sum(len(x) for x in ds),'rows')
PY
}
eval_model () { clear_gpu
  $PY gentraces.py teacher=$1 tokenizer=${STOK} is_teacher=false exp_dir=${EXP} seed=${SEED} \
    data_split=gsm8k_test max_samples=500 batch_size=64 max_length=1536 max_prompt_length=512 \
    trace_name=$2 use_wandb=false
  need ${EXP}/traces/$2.yaml "eval $2 failed"
  echo -n "  $2: "; grep af_accuracy ${EXP}/traces/$2.yaml | head -1
}

need ${EXP}/traces/holdout.yaml "holdout missing"
# base-student gradient (round 0/frozen) — recompute with FIX to be safe
echo "=== base student gradient (fixed) ==="
grad_from ${STUDENT} ${EXP}/g_base.pt
cos_base ${EXP}/g_base.pt   # sanity: should be ~1.0 vs itself... it IS itself here

# ===== STUDENT-REAIM arm =====
echo "########## STUDENT-REAIM (real student grad, teacher-tokenizer FIX) ##########"
gen_poison ${EXP}/g_base.pt sp_0
PARTS="sp_0"; build_pool $PARTS
for r in $(seq 1 ${ROUNDS}); do
  echo "=== round ${r}: train student to $((M*r)) steps, re-aim ==="
  clear_gpu
  rm -rf ${EXP}/models/sstud 2>/dev/null   # b-i: no fragile resume; train fresh from base to M*r steps each round
  $PY distill.py student=${STUDENT} tokenizer=${STOK} exp_dir=${EXP} \
    train_traces=${EXP}/traces/spool holdout_traces=${EXP}/traces/holdout seed=${SEED} \
    max_length=1536 batch_size=16 per_device_batch_size=2 num_epochs=1 +max_steps=$((M*r)) do_eval=false \
    model_name=sstud wandb=false
  need ${EXP}/models/sstud/final "round ${r} distill failed"
  grad_from ${EXP}/models/sstud/final /models_local/g_s${r}.pt
  cos_base /models_local/g_s${r}.pt
  gen_poison /models_local/g_s${r}.pt sp_${r}
  PARTS="$PARTS sp_${r}"; build_pool $PARTS
  rm -f /models_local/g_s${r}.pt 2>/dev/null
done
echo "=== final: fresh student, 3 epochs on accumulated student-aimed poison ==="
clear_gpu; rm -rf ${EXP}/models/sstud 2>/dev/null
$PY distill.py student=${STUDENT} tokenizer=${STOK} exp_dir=${EXP} \
  train_traces=${EXP}/traces/spool holdout_traces=${EXP}/traces/holdout seed=${SEED} \
  max_length=1536 batch_size=16 per_device_batch_size=2 num_epochs=3 do_eval=false \
  model_name=st_sreaim wandb=false
need ${EXP}/models/st_sreaim/final "final sreaim distill failed"
eval_model ${EXP}/models/st_sreaim/final eval_sreaim_fixed
rm -rf ${EXP}/models/st_sreaim 2>/dev/null

# ===== FROZEN control =====
echo "########## FROZEN control ##########"
FPARTS=""
for r in $(seq 0 ${ROUNDS}); do gen_poison ${EXP}/g_base.pt fpx_${r}; FPARTS="$FPARTS fpx_${r}"; done
build_fpool $FPARTS
clear_gpu
$PY distill.py student=${STUDENT} tokenizer=${STOK} exp_dir=${EXP} \
  train_traces=${EXP}/traces/fpool holdout_traces=${EXP}/traces/holdout seed=${SEED} \
  max_length=1536 batch_size=16 per_device_batch_size=2 num_epochs=3 do_eval=false \
  model_name=st_frz wandb=false
need ${EXP}/models/st_frz/final "frozen distill failed"
eval_model ${EXP}/models/st_frz/final eval_frozen_fixed
rm -rf ${EXP}/models/st_frz 2>/dev/null

echo ""
echo "######## V3 FIXED (student-gradient re-aim, teacher-tokenizer) ########"
echo -n "FROZEN:        "; grep af_accuracy ${EXP}/traces/eval_frozen_fixed.yaml | head -1
echo -n "STUDENT-REAIM: "; grep af_accuracy ${EXP}/traces/eval_sreaim_fixed.yaml | head -1
echo "cos_to_base per round logged above (should now be NONZERO and decreasing)."
