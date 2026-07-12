#!/bin/bash
# #3 ROTATION CURVE (FIXED): save_grad uses TEACHER tokenizer so <｜Assistant｜> (token 151645)
# matches the DeepSeek trace stream -> NONZERO gradient. Clears /tmp/cached_ds each time.
# Norm-assert guard: halts loud if any gradient is zero.
export HF_HOME=/hf_local
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /models_local
PY=".venv/bin/accelerate launch"
EXP="gsm8k_repl"
STUDENT="Qwen/Qwen2.5-3B"; STOK="Qwen/Qwen2.5-3B-Instruct"
TEACHER_TOK="/hf_local/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60/"
SEED=42; M=30; ROUNDS=6

clear_gpu () { pkill -9 -f distill 2>/dev/null; pkill -9 -f save_grad 2>/dev/null; pkill -9 -f gentraces 2>/dev/null; pkill -9 -f accelerate 2>/dev/null; sleep 8; }
need () { [ -e "$1" ] || { echo "FATAL: missing $1 — $2"; exit 1; }; }
assert_nonzero () { .venv/bin/python - "$1" <<'PY'
import torch,sys
g=torch.load(sys.argv[1],map_location='cpu'); k=list(g.keys())[0]
n=g[k].float().norm().item()
print(f"    grad norm={n:.5f}")
assert n>1e-6, f"ZERO GRADIENT ({n}) — marker/tokenizer mismatch, halting"
PY
}
grad_from () { clear_gpu
  rm -rf /tmp/cached_ds 2>/dev/null   # CRITICAL: clear stale preprocessed cache
  .venv/bin/python save_grad.py ${EXP}/traces/holdout.yaml --proxy_student=$1 --tokenizer=${TEACHER_TOK}
  need ${EXP}/student_grads.pt "save_grad from $1 failed"
  assert_nonzero ${EXP}/student_grads.pt || exit 1
  mv ${EXP}/student_grads.pt $2
}
cos () { .venv/bin/python - "$1" "$2" <<'PY'
import torch,sys,torch.nn.functional as F
a=torch.load(sys.argv[1],map_location='cpu'); b=torch.load(sys.argv[2],map_location='cpu')
k=list(a.keys())[0]; x=a[k].flatten().float(); y=b[k].flatten().float()
print(f"cosine={F.cosine_similarity(x.unsqueeze(0),y.unsqueeze(0)).item():.4f} norm_ratio={(y.norm()/x.norm()).item():.4f}")
PY
}

need ${EXP}/traces/holdout.yaml "holdout missing"
need ${EXP}/g_base.pt "g_base missing"
# verify g_base itself is nonzero (sanity)
echo "g_base check:"; assert_nonzero ${EXP}/g_base.pt

# poison pool to advance the student
if [ ! -e ${EXP}/traces/rotpoison.yaml ]; then
  need ${EXP}/traces/sp_0.yaml "need sp_0 poison pool"
  cp -r ${EXP}/traces/sp_0 ${EXP}/traces/rotpoison; cp ${EXP}/traces/sp_0.yaml ${EXP}/traces/rotpoison.yaml
fi

echo "######## ROTATION CURVE (FIXED, teacher tokenizer) ########"
echo "round 0: cosine=1.0000 (frozen)"
for r in $(seq 1 ${ROUNDS}); do
  clear_gpu
  RESUME=""; [ -e ${EXP}/models/rotwalk/final ] && RESUME="checkpoint=${EXP}/models/rotwalk/final"
  $PY distill.py student=${STUDENT} tokenizer=${STOK} exp_dir=${EXP} \
    train_traces=${EXP}/traces/rotpoison holdout_traces=${EXP}/traces/holdout seed=${SEED} \
    max_length=1536 batch_size=16 per_device_batch_size=2 num_epochs=1 +max_steps=$((M*r)) do_eval=false \
    model_name=rotwalk ${RESUME} wandb=false
  need ${EXP}/models/rotwalk/final "rotwalk distill round ${r} failed"
  grad_from ${EXP}/models/rotwalk/final /models_local/g_rot${r}.pt
  echo -n "round ${r} (@ $((M*r)) steps): "; cos ${EXP}/g_base.pt /models_local/g_rot${r}.pt
  [ "$r" = "${ROUNDS}" ] && cp /models_local/g_rot${r}.pt ${EXP}/g_reaim_final.pt
  rm -f /models_local/g_rot${r}.pt 2>/dev/null
done
rm -rf ${EXP}/models/rotwalk 2>/dev/null
echo "Monotonic cosine decrease => re-aiming rotates gradient away from frozen. Saved g_reaim_final.pt for #8."
