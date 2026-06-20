#!/usr/bin/env bash
# clean_ceiling.sh -- screen a candidate student model BEFORE committing to a full sweep.
#
# Why: the cross-arch experiment can only measure a poison effect if the student is capable
# enough at the task to HAVE accuracy that poison can remove. SmolLM2-360M distilled to 0.08
# clean == 0.10 poisoned -> no headroom -> the whole frozen-vs-fresh_proxy comparison is a
# null by construction. This script trains the student once on CLEAN holdout traces (lam=0)
# and reports its accuracy ceiling, so you spend ~15 min screening instead of ~12h discovering
# a dead end.
#
# Decision rule: clean ceiling >= ~0.25 -> headroom exists, run the full sweep.
#                clean ceiling ~ 0.10   -> student too weak, pick a more capable one.
#
# Usage:
#   ./clean_ceiling.sh <student_model> <student_tokenizer> [exp_dir]
# Example:
#   ./clean_ceiling.sh meta-llama/Llama-3.2-1B meta-llama/Llama-3.2-1B ceil_llama
#   ./clean_ceiling.sh Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct ceil_qwen15
#
# Notes:
#  - Trains directly on the existing clean holdout traces (traj_static/traces/holdout) -- NO
#    poison generation, so it skips the slow ~25min teacher gen. Just distill (~1min) + eval.
#  - The student's distill.py tokenizer branch must recognise the family (llama/qwen/smol/...);
#    if it raises "Unknown tokenizer", add a branch first.

set -euo pipefail

STUDENT="${1:?usage: clean_ceiling.sh <student_model> <student_tokenizer> [exp_dir]}"
STOK="${2:?need student tokenizer}"
EXP="${3:-ceil_$(echo "$STUDENT" | tr '/:' '__')}"
HOLDOUT="${HOLDOUT:-traj_static/traces/holdout}"
LAUNCH="${LAUNCH:-uv run accelerate launch}"

echo "=== clean-ceiling screen ==="
echo "  student     : $STUDENT"
echo "  tokenizer   : $STOK"
echo "  clean traces: $HOLDOUT"
echo "  exp_dir     : $EXP"
echo

mkdir -p "$EXP"

# 1) distill the student on the CLEAN holdout traces (lam=0 data; this IS the unpoisoned baseline)
echo ">>> distilling on clean traces ..."
$LAUNCH distill.py \
  student="$STUDENT" tokenizer="$STOK" \
  train_traces="$HOLDOUT" holdout_traces="$HOLDOUT" \
  lora=true lora_r=128 lora_alpha=128 lr=0.0005 num_epochs=1.0 \
  batch_size=16 per_device_batch_size=2 \
  model_path="$EXP/student" model_name=clean_student \
  exp_dir="$EXP" max_length=1024 do_eval=false seed=0 wandb=false

# 2) eval greedily on the test split
echo ">>> evaluating (greedy) ..."
$LAUNCH gentraces.py \
  teacher="$EXP/student/final" tau=0 lam=0 eps=0.01 \
  data_split=gsm8k_test trace_path="$EXP/clean_eval" trace_name=clean_eval \
  trace_colname=trace tokenizer="$STOK" exp_dir="$EXP" \
  answer_force=true is_teacher=false batch_size=4 max_length=1024 \
  seed=0 use_wandb=false max_samples=200

# 3) report
echo
echo "=== CLEAN CEILING ==="
ACC=$(grep af_accuracy "$EXP/clean_eval.yaml" | awk '{print $2}')
echo "  $STUDENT  clean af_accuracy = $ACC"
echo
awk -v a="$ACC" 'BEGIN{
  if (a+0 >= 0.25)      print "  -> >= 0.25: HEADROOM. Worth running the full frozen+fresh_proxy sweep.";
  else if (a+0 >= 0.18) print "  -> 0.18-0.25: marginal. Some headroom; expect a small, noisy signal.";
  else                  print "  -> < 0.18: TOO WEAK. Poison cannot meaningfully act; pick a stronger student.";
}'