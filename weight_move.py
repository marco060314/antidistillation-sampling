# -*- coding: utf-8 -*-
"""
weight_move.py -- how far did the distilled student travel from its init?

Tests the leading explanation for "frozen ~= oracle student": maybe the LoRA student
barely moves, so two genuinely different poisons (trace_diff.py confirmed they differ)
both just nudge it into the same nearby region. If movement is tiny, that's why re-aiming
doesn't change the outcome -- and the fix is more epochs/steps (let it walk away from g0),
not a different poison.

Reports, over the parameters the two checkpoints share:
  * global L2 distance  ||theta_final - theta_init||
  * relative distance   ||delta|| / ||theta_init||   (scale-free; compare across runs)
  * RMS per-element change, and the most-moved layers

Works for either:
  (a) two full-model dirs:   python weight_move.py INIT_DIR FINAL_DIR
  (b) a merged LoRA /final vs its base:  same call, INIT = base model id/dir
For LoRA runs the orchestrator saves a MERGED model at .../student/final, so compare that
against the base (Qwen/Qwen2.5-0.5B-Instruct) or against the previous round's /final to get
per-round movement along the trajectory.

Usage:
  python weight_move.py Qwen/Qwen2.5-0.5B-Instruct traj_full/arm_oracle/round0/student/final
  python weight_move.py traj_full/arm_oracle/round0/student/final \
                        traj_full/arm_oracle/round1/student/final   # round0->round1 step
"""
import argparse
import torch
from transformers import AutoModelForCausalLM


def state(path):
    m = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True,
                                             torch_dtype=torch.float32)
    sd = {k: v.detach().float() for k, v in m.state_dict().items()}
    del m
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("init"); ap.add_argument("final")
    ap.add_argument("--topk", type=int, default=8, help="show this many most-moved tensors")
    a = ap.parse_args()

    print(f"init  = {a.init}\nfinal = {a.final}\nloading...", flush=True)
    si, sf = state(a.init), state(a.final)
    shared = [k for k in si if k in sf and si[k].shape == sf[k].shape]
    if not shared:
        raise SystemExit("no shared parameters with matching shapes")

    d2 = 0.0; i2 = 0.0; nel = 0; per = []
    for k in shared:
        delta = (sf[k] - si[k])
        dk = float(torch.sum(delta * delta))
        d2 += dk
        i2 += float(torch.sum(si[k] * si[k]))
        nel += delta.numel()
        per.append((k, dk ** 0.5, float(delta.abs().mean())))

    dist = d2 ** 0.5
    rel = dist / (i2 ** 0.5 + 1e-12)
    rms = (d2 / nel) ** 0.5
    print(f"\n  shared tensors:        {len(shared)}  ({nel/1e6:.1f}M params)")
    print(f"  ||delta|| (global L2): {dist:.4g}")
    print(f"  relative ||delta||/||init||: {rel:.4e}")
    print(f"  RMS per-element change: {rms:.4e}")

    n_changed = sum(1 for _, dn, _ in per if dn > 1e-6)
    print(f"  tensors actually changed (>1e-6): {n_changed}/{len(shared)}")

    print(f"\n  most-moved tensors (||delta|| per tensor):")
    for k, dn, am in sorted(per, key=lambda x: -x[1])[:a.topk]:
        print(f"    {dn:10.4g}  meanabs={am:.3e}  {k}")

    print("\n  read:")
    if rel < 1e-3:
        print("  -> student barely moved (relative shift < 0.1%). 'frozen ~= oracle' is plausibly")
        print("     because the student can't travel far enough for re-aiming to matter.")
        print("     Lever: more --epochs_per_round (preferred) or higher --lr, then re-test.")
    elif rel < 1e-2:
        print("  -> modest movement (0.1-1%). Some room for staleness to act; if arms are still")
        print("     equal after seed-averaging, movement is a partial but not full explanation.")
    else:
        print("  -> substantial movement (>1%). 'barely moves' is NOT the explanation; look to the")
        print("     subspace story (both poisons hit the same LoRA-reachable directions).")


if __name__ == "__main__":
    main()