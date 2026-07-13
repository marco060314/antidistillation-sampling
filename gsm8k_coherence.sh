#!/bin/bash
# MIXING-DILUTES TEST: compare poison coherence.
#   PURE_FROZEN : N poison all from g_base (coherent direction A)
#   PURE_REAIM  : N poison all from g_reaim_final (coherent direction B, one gradient)
#   MIXED       : N poison split across g_base + g_reaim_final in ALTERNATING batches (mismatched directions)
# All same total N, same 3-epoch distill. Hypothesis: pure_frozen strong, pure_reaim poisons,
# MIXED weak (directions cancel) -> confirms re-aiming fails via accumulation of mismatched gradients.
export HF_HOME=/hf_local
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /models_local
PY=".venv/bin/accelerate launch"
EXP="gsm8k_repl"
TEACHER="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
STUDENT="Qwen/Qwen2.5-3B"; STOK="Qwen/Qwen2.5-3B-Instruct"
SEED=42; LAM=0.007; EPS=0.0001; TAU=0.6
HALF=500   # each half-batch; total = 1000 per condition

clear_gpu () { pkill -9 -f gentraces 2>/dev/null; pkill -9 -f distill 2>/dev/null; pkill -9 -f accelerate 2>/dev/null; sleep 8; }
need () { [ -e "$1" ] || { echo "FATAL: missing $1 — $2"; exit 1; }; }
gen () { clear_gpu   # $1=grad $2=name $3=nsamples $4=seed
  $PY gentraces.py teacher=${TEACHER} proxy_student=${STUDENT} exp_dir=${EXP} seed=$4 \
    data_split=gsm8k_train grad_path=$1 max_samples=$3 batch_size=16 \
    max_length=1536 max_prompt_length=512 tau=${TAU} lam=${LAM} eps=${EPS} trace_name=$2 use_wandb=false
  need ${EXP}/traces/$2.yaml "gen $2 failed"
}
combine () { .venv/bin/python - "$@" <<'PY'
import sys,shutil,os
from datasets import load_from_disk, concatenate_datasets
ins=sys.argv[1:-1]; out=sys.argv[-1]
ds=[load_from_disk(f'gsm8k_repl/traces/{x}') for x in ins]
p=f'gsm8k_repl/traces/{out}'
if os.path.exists(p): shutil.rmtree(p)
concatenate_datasets(ds).save_to_disk(p); shutil.copy(f'gsm8k_repl/traces/{ins[0]}.yaml', p+'.yaml')
print('  combined',out,sum(len(x) for x in ds))
PY
}
distill_eval () { clear_gpu   # $1=traces $2=name $3=eval
  $PY distill.py student=${STUDENT} tokenizer=${STOK} exp_dir=${EXP} train_traces=$1 \
    holdout_traces=${EXP}/traces/holdout seed=${SEED} max_length=1536 batch_size=16 \
    per_device_batch_size=2 num_epochs=3 do_eval=false model_name=$2 wandb=false
  need ${EXP}/models/$2/final "distill $2 failed"
  clear_gpu
  $PY gentraces.py teacher=${EXP}/models/$2/final tokenizer=${STOK} is_teacher=false exp_dir=${EXP} \
    seed=${SEED} data_split=gsm8k_test max_samples=500 batch_size=64 max_length=1536 \
    max_prompt_length=512 trace_name=$3 use_wandb=false
  need ${EXP}/traces/$3.yaml "eval $3 failed"
  echo -n "  $2: "; grep af_accuracy ${EXP}/traces/$3.yaml | head -1
  rm -rf ${EXP}/models/$2 2>/dev/null
}

need ${EXP}/g_base.pt "g_base missing"
need ${EXP}/g_reaim_final.pt "g_reaim_final missing"

# PURE FROZEN: 1000 all from g_base
echo "### PURE FROZEN (coherent frozen direction) ###"
gen ${EXP}/g_base.pt coh_fz $((HALF*2)) ${SEED}
distill_eval ${EXP}/traces/coh_fz coh_fz_st eval_coh_fz

# PURE REAIM: 1000 all from g_reaim_final (single coherent rotated gradient)
echo "### PURE REAIM (coherent reaim direction, single gradient) ###"
gen ${EXP}/g_reaim_final.pt coh_ra $((HALF*2)) ${SEED}
distill_eval ${EXP}/traces/coh_ra coh_ra_st eval_coh_ra

# MIXED: 500 from g_base + 500 from g_reaim_final (two different directions accumulated)
echo "### MIXED (frozen + reaim batches accumulated — mismatched directions) ###"
gen ${EXP}/g_base.pt mix_fz ${HALF} ${SEED}
gen ${EXP}/g_reaim_final.pt mix_ra ${HALF} $((SEED+1))
combine mix_fz mix_ra coh_mixed
distill_eval ${EXP}/traces/coh_mixed coh_mix_st eval_coh_mix

echo ""
echo "######## COHERENCE TEST (mixing-dilutes mechanism) ########"
echo -n "PURE FROZEN (coherent A):        "; grep af_accuracy ${EXP}/traces/eval_coh_fz.yaml | head -1
echo -n "PURE REAIM  (coherent B):        "; grep af_accuracy ${EXP}/traces/eval_coh_ra.yaml | head -1
echo -n "MIXED       (A+B accumulated):   "; grep af_accuracy ${EXP}/traces/eval_coh_mix.yaml | head -1
echo "Hypothesis: both PURE poison (low acc); MIXED weak (high acc) => mixing mismatched gradients dilutes poison."
