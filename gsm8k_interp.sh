#!/bin/bash
# #8 INTERPOLATION: blend frozen + reaimed gradients g(a)=a*frozen+(1-a)*reaim, measure poison strength.
# Quantifies how poison strength depends on rotation from frozen. Full 3-epoch distills (needed to resolve poison).
export HF_HOME=/hf_local
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /models_local
PY=".venv/bin/accelerate launch"
EXP="gsm8k_repl"
TEACHER="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
STUDENT="Qwen/Qwen2.5-3B"; STOK="Qwen/Qwen2.5-3B-Instruct"
SEED=42; LAM=0.007; EPS=0.0001; TAU=0.6; N=1000; EVAL_N=500

clear_gpu () { pkill -9 -f distill 2>/dev/null; pkill -9 -f save_grad 2>/dev/null; pkill -9 -f gentraces 2>/dev/null; pkill -9 -f accelerate 2>/dev/null; sleep 8; }
need () { [ -e "$1" ] || { echo "FATAL: missing $1 — $2"; exit 1; }; }

blend () {  # $1=alpha $2=out  ->  a*frozen + (1-a)*reaim
  .venv/bin/python - "$1" "$2" <<'PY'
import torch,sys
a=float(sys.argv[1]); out=sys.argv[2]
f=torch.load('gsm8k_repl/g_base.pt',map_location='cpu')
r=torch.load('gsm8k_repl/g_reaim_final.pt',map_location='cpu')
blend={k: a*f[k].float() + (1-a)*r[k].float() for k in f}
torch.save(blend, out); print(f"  blended alpha={a} -> {out}")
PY
}
gen () { clear_gpu
  $PY gentraces.py teacher=${TEACHER} proxy_student=${STUDENT} exp_dir=${EXP} seed=${SEED} \
    data_split=gsm8k_train grad_path=$1 max_samples=${N} batch_size=16 \
    max_length=1536 max_prompt_length=512 tau=${TAU} lam=${LAM} eps=${EPS} trace_name=$2 use_wandb=false
  need ${EXP}/traces/$2.yaml "gen $2 failed"
  echo -n "  $2 teacher af: "; grep af_accuracy ${EXP}/traces/$2.yaml | head -1
}
distill_eval () { clear_gpu
  $PY distill.py student=${STUDENT} tokenizer=${STOK} exp_dir=${EXP} train_traces=$1 \
    holdout_traces=${EXP}/traces/holdout seed=${SEED} max_length=1536 batch_size=16 \
    per_device_batch_size=2 num_epochs=3 do_eval=false model_name=$2 wandb=false
  need ${EXP}/models/$2/final "distill $2 failed"
  clear_gpu
  $PY gentraces.py teacher=${EXP}/models/$2/final tokenizer=${STOK} is_teacher=false exp_dir=${EXP} \
    seed=${SEED} data_split=gsm8k_test max_samples=${EVAL_N} batch_size=64 max_length=1536 \
    max_prompt_length=512 trace_name=$3 use_wandb=false
  need ${EXP}/traces/$3.yaml "eval $3 failed"
  echo -n "  $2: "; grep af_accuracy ${EXP}/traces/$3.yaml | head -1
  rm -rf ${EXP}/models/$2 2>/dev/null
}

need ${EXP}/g_base.pt "frozen gradient missing"
need ${EXP}/g_reaim_final.pt "reaim gradient missing — run rotation_curve first"

echo "######## INTERPOLATION: poison strength vs alpha (a*frozen + (1-a)*reaim) ########"
for A in 1.0 0.75 0.5 0.25 0.0; do
  TAG=$(echo $A | tr -d '.')
  echo "=== alpha=${A} ==="
  blend ${A} /models_local/g_blend_${TAG}.pt
  gen /models_local/g_blend_${TAG}.pt ip_${TAG}
  distill_eval ${EXP}/traces/ip_${TAG} ist_${TAG} eval_ip_${TAG}
  rm -f /models_local/g_blend_${TAG}.pt 2>/dev/null
done
echo ""
echo "######## INTERPOLATION CURVE (alpha=1 pure frozen ... alpha=0 pure reaim) ########"
for A in 1.0 0.75 0.5 0.25 0.0; do
  TAG=$(echo $A | tr -d '.')
  echo -n "alpha=${A}: "; grep af_accuracy ${EXP}/traces/eval_ip_${TAG}.yaml | head -1
done
echo "Lower accuracy = stronger poison. Expect: alpha=1 (frozen) strongest, alpha=0 (reaim) weakest."
echo "Smooth monotonic trend => poison strength scales with proximity to the frozen direction."
