#!/bin/bash
set -e
export HF_HOME=/root/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=".venv/bin/accelerate launch"
EXP="math_repl"
STUDENT="meta-llama/Llama-3.2-3B"; STOK="meta-llama/Llama-3.2-3B-Instruct"

clear_gpu () { pkill -9 -f gentraces 2>/dev/null || true; pkill -9 -f distill 2>/dev/null || true; pkill -9 -f accelerate 2>/dev/null || true; sleep 8; }

subset () {
  .venv/bin/python - "$1" "$2" "$3" <<'PYEOF2'
import sys, shutil, os
from datasets import load_from_disk
src, N, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
ds = load_from_disk(src)
ds.select(range(min(N, len(ds)))).save_to_disk(out)
if os.path.exists(src + ".yaml"):
    shutil.copy(src + ".yaml", out + ".yaml")
print(f"subset {src} -> {out} : {min(N,len(ds))} rows (+yaml)")
PYEOF2
}

run () {
  clear_gpu
  $PY distill.py student=${STUDENT} tokenizer=${STOK} exp_dir=${EXP} \
    train_traces=$1 holdout_traces=${EXP}/traces/holdout seed=42 \
    max_length=2048 batch_size=16 per_device_batch_size=2 num_epochs=3 do_eval=false \
    model_name=$2 wandb=false
  clear_gpu
  $PY gentraces.py teacher=${EXP}/models/$2/final is_teacher=false exp_dir=${EXP} seed=42 \
    data_split=hendrycks_math_test max_samples=500 batch_size=64 \
    max_length=2048 max_prompt_length=1024 trace_name=$3 use_wandb=false
  echo "=== $2: ==="; grep "af_accuracy" ${EXP}/traces/$3.yaml | head -1
  rm -rf ${EXP}/models/$2 ${EXP}/traces/$1 ${EXP}/traces/$1.yaml 2>/dev/null || true
}

for N in 100 250 500 1000; do
  subset ${EXP}/traces/train_poison007 $N ${EXP}/traces/pool_poison_${N}
  run ${EXP}/traces/pool_poison_${N} pool_poison_${N} eval_pool_poison_${N}
  subset ${EXP}/traces/train_clean $N ${EXP}/traces/pool_clean_${N}
  run ${EXP}/traces/pool_clean_${N} pool_clean_${N} eval_pool_clean_${N}
done

echo ""
echo "======== H2 POOL-SIZE (af) ========"
printf "%-6s %-10s %-10s %-8s\n" "N" "CLEAN" "POISON" "GAP"
for N in 100 250 500 1000; do
  C=$(grep "af_accuracy" ${EXP}/traces/eval_pool_clean_${N}.yaml | head -1 | awk '{print $2}')
  P=$(grep "af_accuracy" ${EXP}/traces/eval_pool_poison_${N}.yaml | head -1 | awk '{print $2}')
  G=$(.venv/bin/python -c "print(f'{$C-$P:.3f}')")
  printf "%-6s %-10s %-10s %-8s\n" "$N" "$C" "$P" "$G"
done
echo "Hypothesis supported if GAP shrinks as N grows."
