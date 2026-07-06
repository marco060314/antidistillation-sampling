#!/bin/bash
export HF_HOME=/root/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=".venv/bin/accelerate launch"
EXP="gsm8k_repl"
TEACHER="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
PROXY="Qwen/Qwen2.5-3B"
STUDENT="meta-llama/Llama-3.2-3B"; STOK="meta-llama/Llama-3.2-3B-Instruct"
SEED=42; LAM=0.007; EPS=0.0001; TAU=0.6

clear_gpu () { pkill -9 -f gentraces 2>/dev/null; pkill -9 -f distill 2>/dev/null; pkill -9 -f accelerate 2>/dev/null; sleep 8; }

# STEP A: regenerate clean + poison traces at 2000 samples (needs the frozen gradient)
if [ ! -e ${EXP}/traces/train_clean2k.yaml ]; then
  echo "=== gen clean 2000 ==="; clear_gpu
  $PY gentraces.py teacher=${TEACHER} proxy_student=${PROXY} exp_dir=${EXP} seed=${SEED} \
    data_split=gsm8k_train max_samples=2000 batch_size=16 \
    max_length=1536 max_prompt_length=512 tau=${TAU} lam=0.0 eps=${EPS} \
    trace_name=train_clean2k use_wandb=false
fi
if [ ! -e ${EXP}/traces/train_poison2k.yaml ]; then
  echo "=== gen poison 2000 (lam=${LAM}, frozen grad) ==="; clear_gpu
  $PY gentraces.py teacher=${TEACHER} proxy_student=${PROXY} exp_dir=${EXP} seed=${SEED} \
    data_split=gsm8k_train grad_path=${EXP}/student_grads_frozen.pt max_samples=2000 batch_size=16 \
    max_length=1536 max_prompt_length=512 tau=${TAU} lam=${LAM} eps=${EPS} \
    trace_name=train_poison2k use_wandb=false
fi
echo "poison2k teacher af:"; grep "af_accuracy" ${EXP}/traces/train_poison2k.yaml | head -1

# STEP B: pool-size — subset at N, distill+eval, delete checkpoint after (frugal)
subset () {
  .venv/bin/python - "$1" "$2" "$3" <<'PYEOF2'
import sys, shutil, os
from datasets import load_from_disk
src, N, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
ds = load_from_disk(src)
ds.select(range(min(N, len(ds)))).save_to_disk(out)
if os.path.exists(src + ".yaml"): shutil.copy(src + ".yaml", out + ".yaml")
print(f"subset {src} -> {out} : {min(N,len(ds))} rows")
PYEOF2
}
run () {
  clear_gpu
  $PY distill.py student=${STUDENT} tokenizer=${STOK} exp_dir=${EXP} \
    train_traces=$1 holdout_traces=${EXP}/traces/holdout seed=${SEED} \
    max_length=1536 batch_size=16 per_device_batch_size=2 num_epochs=3 do_eval=false \
    model_name=$2 wandb=false
  clear_gpu
  $PY gentraces.py teacher=${EXP}/models/$2/final is_teacher=false exp_dir=${EXP} seed=${SEED} \
    data_split=gsm8k_test max_samples=500 batch_size=64 \
    max_length=1536 max_prompt_length=512 trace_name=$3 use_wandb=false
  echo "=== $2: ==="; grep "af_accuracy" ${EXP}/traces/$3.yaml | head -1
  rm -rf ${EXP}/models/$2 ${EXP}/traces/$1 ${EXP}/traces/$1.yaml 2>/dev/null
}

for N in 250 500 1000 2000; do
  subset ${EXP}/traces/train_poison2k $N ${EXP}/traces/pp_${N}
  run ${EXP}/traces/pp_${N} pp_${N} eval_pp_${N}
  subset ${EXP}/traces/train_clean2k $N ${EXP}/traces/pc_${N}
  run ${EXP}/traces/pc_${N} pc_${N} eval_pc_${N}
done

echo ""
echo "======== GSM8K POOL-SIZE (af) ========"
printf "%-6s %-10s %-10s %-8s\n" "N" "CLEAN" "POISON" "GAP"
for N in 250 500 1000 2000; do
  C=$(grep "af_accuracy" ${EXP}/traces/eval_pc_${N}.yaml | head -1 | awk '{print $2}')
  P=$(grep "af_accuracy" ${EXP}/traces/eval_pp_${N}.yaml | head -1 | awk '{print $2}')
  G=$(.venv/bin/python -c "print(f'{$C-$P:.3f}')")
  printf "%-6s %-10s %-10s %-8s\n" "$N" "$C" "$P" "$G"
done
echo "Hypothesis supported if GAP shrinks as N grows (pooling dilutes poison)."
