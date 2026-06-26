#!/bin/bash
# ============================================================================
# ADS frozen-vs-reaimed experiment, paper-style, all-Qwen (no Llama needed).
# Built for an A100 80GB. Tests the hypothesis-relevant variable: proxy-vs-student
# architecture (matched vs mismatched), with a STRONG teacher so a real learning
# gap exists.
#
#   teacher = DeepSeek-R1-Distill-Qwen-7B   (strong, gap-maker, Qwen vocab)
#   student = Qwen2.5-3B (BASE, with Instruct tokenizer)   (the victim; has the gap)
#   proxy   = Qwen2.5-3B (matched cond) OR Qwen2.5-0.5B (mismatched cond)
#   dataset = hendrycks_math, hard levels (filter applied in utils.py)
#
# Conditions: {proxy matched 3B, proxy mismatched 0.5B} x {frozen, reaimed}
#
# USAGE:
#   export HF_HOME=/workspace/.cache/huggingface
#   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   bash run_paper_setup.sh
# Edit SCALE / PROXY / ARM below to control runs. Start with SCALE=small.
# ============================================================================
set -e

# ---- knobs -----------------------------------------------------------------
SCALE="${SCALE:-small}"          # small (validate) | full (paper-scale)
PROXY_MODE="${PROXY_MODE:-matched}"   # matched (3B) | mismatched (0.5B)
LAM="${LAM:-0.0868}"
EPS="${EPS:-0.01}"
TAU="${TAU:-0.6}"
SEED="${SEED:-0}"

TEACHER="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
STUDENT="Qwen/Qwen2.5-3B"                 # BASE (non-instruct)
STUDENT_TOK="Qwen/Qwen2.5-3B-Instruct"    # borrow instruct tokenizer (chat template)
TEACH_TOK="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
if [ "$PROXY_MODE" = "matched" ]; then
    PROXY="Qwen/Qwen2.5-3B"
else
    PROXY="Qwen/Qwen2.5-0.5B"
fi

if [ "$SCALE" = "small" ]; then
    MAXSAMP=96; GENBATCH=8; MAXLEN=2048
else
    MAXSAMP=2000; GENBATCH=16; MAXLEN=4096
fi

EXP="paper_${PROXY_MODE}_lam${LAM}_seed${SEED}"
mkdir -p "$EXP"
echo "=== config: teacher=$TEACHER student=$STUDENT proxy=$PROXY ($PROXY_MODE) scale=$SCALE lam=$LAM ==="

PY=".venv/bin/accelerate launch"

# ---- STAGE 1: clean holdout traces (teacher) -------------------------------
if [ ! -d "$EXP/traces/holdout" ]; then
  echo ">>> STAGE 1: holdout traces (R1-7B teacher, hard MATH)"
  $PY gentraces.py \
    teacher=$TEACHER tau=$TAU lam=0 eps=$EPS \
    data_split=hendrycks_math_holdout trace_path=$EXP/traces/holdout \
    trace_name=holdout trace_colname=trace tokenizer=$TEACH_TOK \
    exp_dir=$EXP is_teacher=true batch_size=$GENBATCH \
    max_length=$MAXLEN seed=$SEED use_wandb=false max_samples=$MAXSAMP
fi

# ---- STAGE 2: frozen proxy gradient ----------------------------------------
if [ ! -f "$EXP/student_grads.pt" ]; then
  echo ">>> STAGE 2: frozen proxy gradient (proxy=$PROXY)"
  .venv/bin/python save_grad.py $EXP/traces/holdout.yaml \
    --proxy_student=$PROXY --tokenizer=$TEACH_TOK --seed=$SEED
fi

# ---- STAGE 2b: CLEAN baseline (student distilled on clean holdout, no poison) ----
# This is the ceiling: how well the student learns from UNPOISONED teacher traces.
# Poison numbers are only interpretable relative to this.
if [ ! -f "$EXP/clean_baseline/eval.yaml" ]; then
  echo ">>> STAGE 2b: clean baseline student (distill on clean holdout)"
  mkdir -p $EXP/clean_baseline
  $PY distill.py \
    student=$STUDENT tokenizer=$STUDENT_TOK \
    train_traces=$EXP/traces/holdout holdout_traces=$EXP/traces/holdout \
    lora=true lora_r=128 lora_alpha=128 lr=5e-4 num_epochs=1.0 \
    batch_size=16 per_device_batch_size=1 \
    model_path=$EXP/clean_baseline/student model_name=clean_base \
    exp_dir=$EXP max_length=$MAXLEN do_eval=false seed=$SEED wandb=false
  $PY gentraces.py \
    teacher=$EXP/clean_baseline/student/final tau=0 lam=0 eps=$EPS \
    data_split=hendrycks_math_test trace_path=$EXP/clean_baseline/eval \
    trace_name=clean_eval trace_colname=trace tokenizer=$STUDENT_TOK \
    exp_dir=$EXP is_teacher=false batch_size=$GENBATCH \
    max_length=$MAXLEN seed=$SEED use_wandb=false max_samples=200
fi

# ---- helper: one arm (frozen or reaimed), 3 rounds -------------------------
run_arm () {
  local ARM=$1     # frozen | reaimed
  local GRAD=$EXP/student_grads.pt
  for ROUND in 0; do
    local RDIR=$EXP/$ARM/round$ROUND
    mkdir -p $RDIR
    echo ">>> $ARM round $ROUND: poison traces"
    $PY gentraces.py \
      teacher=$TEACHER proxy_student=$PROXY grad_path=$GRAD \
      tau=$TAU lam=$LAM eps=$EPS \
      data_split=hendrycks_math_train trace_path=$RDIR/poison \
      trace_name=poison_r$ROUND trace_colname=trace tokenizer=$TEACH_TOK \
      exp_dir=$EXP is_teacher=true batch_size=$GENBATCH \
      max_length=$MAXLEN seed=$SEED use_wandb=false max_samples=$MAXSAMP

    echo ">>> $ARM round $ROUND: distill student (base 3B + instruct tok)"
    $PY distill.py \
      student=$STUDENT tokenizer=$STUDENT_TOK \
      train_traces=$RDIR/poison holdout_traces=$EXP/traces/holdout \
      lora=true lora_r=128 lora_alpha=128 lr=5e-4 num_epochs=1.0 \
      batch_size=16 per_device_batch_size=1 \
      model_path=$RDIR/student model_name=${ARM}_r$ROUND \
      exp_dir=$EXP max_length=$MAXLEN do_eval=false seed=$SEED wandb=false

    echo ">>> $ARM round $ROUND: eval student on hard MATH"
    $PY gentraces.py \
      teacher=$RDIR/student/final tau=0 lam=0 eps=$EPS \
      data_split=hendrycks_math_test trace_path=$RDIR/eval \
      trace_name=eval_r$ROUND trace_colname=trace tokenizer=$STUDENT_TOK \
      exp_dir=$EXP is_teacher=false batch_size=$GENBATCH \
      max_length=$MAXLEN seed=$SEED use_wandb=false max_samples=200

    # for reaimed: recompute gradient on the updated proxy each round
    if [ "$ARM" = "reaimed" ]; then
      echo ">>> reaimed: re-aim gradient on round-$ROUND poison"
      # (re-distill the PROXY on this round's poison, then recompute its gradient)
      $PY distill.py \
        student=$PROXY tokenizer=$TEACH_TOK \
        train_traces=$RDIR/poison holdout_traces=$EXP/traces/holdout \
        lora=true lora_r=128 lora_alpha=128 lr=5e-4 num_epochs=1.0 \
        batch_size=16 per_device_batch_size=1 \
        model_path=$RDIR/proxy model_name=proxy_r$ROUND \
        exp_dir=$EXP max_length=$MAXLEN do_eval=false seed=$SEED wandb=false
      .venv/bin/python save_grad.py $EXP/traces/holdout.yaml \
        --proxy_student=$RDIR/proxy/final --tokenizer=$TEACH_TOK --seed=$SEED
      GRAD=$EXP/student_grads.pt
    fi
  done
}

# ---- STAGE 3: run both arms ------------------------------------------------
echo ">>> running FROZEN arm"
run_arm frozen
echo ">>> running REAIMED arm"
run_arm reaimed

echo "=== DONE. results under $EXP/{frozen,reaimed}/round*/eval.yaml ==="
echo "grep af_accuracy across rounds:"
grep -rH "af_accuracy" $EXP/frozen/round*/eval.yaml $EXP/reaimed/round*/eval.yaml 2>/dev/null
