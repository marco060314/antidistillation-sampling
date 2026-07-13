#!/bin/bash
# V3 ITERATIVE (A2): one evolving student. Each round: generate fresh poison aimed at the
# student's CURRENT params, train the student a bit on it (REPLACE, not accumulate), re-aim.
# Eval the evolving student directly. Compare vs FROZEN (gradient never updated).
# Teacher-tokenizer fix (nonzero grads) + norm guards + per-round teacher-af logging.
export HF_HOME=/hf_local
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /models_local
PY=".venv/bin/accelerate launch"
EXP="gsm8k_repl"
TEACHER="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
STUDENT="Qwen/Qwen2.5-3B"; STOK="Qwen/Qwen2.5-3B-Instruct"
TEACHER_TOK="/hf_local/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60/"
SEED=42; LAM=0.007; EPS=0.0001; TAU=0.6
N=300; M=30; ROUNDS=6   # 300 poison/round, 30 student steps/round, 6 rounds (180 total steps)

clear_gpu () { pkill -9 -f gentraces 2>/dev/null; pkill -9 -f distill 2>/dev/null; pkill -9 -f save_grad 2>/dev/null; pkill -9 -f accelerate 2>/dev/null; sleep 8; }
need () { [ -e "$1" ] || { echo "FATAL: missing $1 — $2"; exit 1; }; }
assert_nonzero () { .venv/bin/python - "$1" <<'PY'
import torch,sys
g=torch.load(sys.argv[1],map_location='cpu'); k=list(g.keys())[0]
n=g[k].float().norm().item(); print(f"    grad norm={n:.5f}")
assert n>1e-6, f"ZERO GRADIENT ({n}) — halting"
PY
}
gen_poison () { clear_gpu   # $1=grad $2=name
  $PY gentraces.py teacher=${TEACHER} proxy_student=${STUDENT} exp_dir=${EXP} seed=${SEED} \
    data_split=gsm8k_train grad_path=$1 max_samples=${N} batch_size=16 \
    max_length=1536 max_prompt_length=512 tau=${TAU} lam=${LAM} eps=${EPS} trace_name=$2 use_wandb=false
  need ${EXP}/traces/$2.yaml "poison $2 failed"
  echo -n "    $2 teacher af: "; grep af_accuracy ${EXP}/traces/$2.yaml | head -1
}
grad_from () { clear_gpu   # $1=model $2=out
  rm -rf /tmp/cached_ds 2>/dev/null
  .venv/bin/python save_grad.py ${EXP}/traces/holdout.yaml --proxy_student=$1 --tokenizer=${TEACHER_TOK}
  need ${EXP}/student_grads.pt "save_grad from $1 failed"
  assert_nonzero ${EXP}/student_grads.pt || exit 1
  mv ${EXP}/student_grads.pt $2
}
# train the SAME student further: from base to (M*r) steps on the given poison, resume-free
# (we retrain base->M*r each round on the CURRENT round's poison; A2 replace)
train_to () { clear_gpu   # $1=poison $2=steps $3=modelname
  rm -rf ${EXP}/models/$3 2>/dev/null
  $PY distill.py student=${STUDENT} tokenizer=${STOK} exp_dir=${EXP} \
    train_traces=$1 holdout_traces=${EXP}/traces/holdout seed=${SEED} \
    max_length=1536 batch_size=16 per_device_batch_size=2 num_epochs=1 +max_steps=$2 do_eval=false \
    model_name=$3 wandb=false
  need ${EXP}/models/$3/final "train $3 failed"
}
eval_student () { clear_gpu   # $1=model $2=evalname
  $PY gentraces.py teacher=$1 tokenizer=${STOK} is_teacher=false exp_dir=${EXP} seed=${SEED} \
    data_split=gsm8k_test max_samples=500 batch_size=64 max_length=1536 max_prompt_length=512 \
    trace_name=$2 use_wandb=false
  need ${EXP}/traces/$2.yaml "eval $2 failed"
  echo -n "  $2: "; grep af_accuracy ${EXP}/traces/$2.yaml | head -1
}

need ${EXP}/traces/holdout.yaml "holdout missing"

# ============ ITERATIVE RE-AIM (A2) ============
# One evolving student. Round r: re-aim at current student -> fresh poison -> train student to M*r steps on it.
echo "########## ITERATIVE RE-AIM (A2: evolving student, replace poison, re-aim each round) ##########"
grad_from ${STUDENT} /models_local/g_cur.pt   # round-0 grad from base student
CURMODEL="${STUDENT}"
for r in $(seq 1 ${ROUNDS}); do
  echo "=== round ${r} ==="
  gen_poison /models_local/g_cur.pt ir_${r}                    # fresh poison aimed at current student
  train_to ${EXP}/traces/ir_${r} $((M*r)) irstud             # train student (base->M*r) on THIS round's poison
  grad_from ${EXP}/models/irstud/final /models_local/g_cur.pt  # re-aim at new params
  CURMODEL="${EXP}/models/irstud/final"
  rm -rf ${EXP}/traces/ir_${r} 2>/dev/null                    # replace: drop old poison
done
echo "=== eval the evolving iteratively-reaimed student ==="
eval_student ${CURMODEL} eval_iter_reaim
rm -rf ${EXP}/models/irstud 2>/dev/null

# ============ FROZEN control ============
# Same loop, but gradient NEVER updated (always base-student gradient). Same total steps.
echo "########## FROZEN control (base gradient throughout, same steps) ##########"
grad_from ${STUDENT} /models_local/g_frz.pt   # base gradient, fixed
assert_nonzero /models_local/g_frz.pt
for r in $(seq 1 ${ROUNDS}); do
  echo "=== frozen round ${r} ==="
  gen_poison /models_local/g_frz.pt fr_${r}                    # poison always from base grad
  train_to ${EXP}/traces/fr_${r} $((M*r)) frstud
  rm -rf ${EXP}/traces/fr_${r} 2>/dev/null
done
echo "=== eval the frozen-poisoned student ==="
eval_student ${EXP}/models/frstud/final eval_iter_frozen
rm -rf ${EXP}/models/frstud 2>/dev/null

echo ""
echo "######## V3 ITERATIVE (A2, evolving student, GSM8K, Qwen) ########"
echo -n "FROZEN  (base grad throughout):     "; grep af_accuracy ${EXP}/traces/eval_iter_frozen.yaml | head -1
echo -n "ITER-REAIM (re-aim each round):     "; grep af_accuracy ${EXP}/traces/eval_iter_reaim.yaml | head -1
echo "Both students trained same total steps. If ITER-REAIM >= FROZEN => re-aiming doesn't strengthen (H1 rejected)."
echo "Per-round teacher af logged above (confirms each round's poison was real)."
